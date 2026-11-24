import subprocess,sys
subprocess.check_call([sys.executable,'-m','pip','install','-q','-U','bitsandbytes>=0.46.1'])
import json,random
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
F=['Date','Mode','Category','Sub category','Transaction Type','Amount']
def examples():
    for kind, noun in [('Income', 'income'), ('Expense', 'spending')]:
        condition = {'column': 'Transaction Type', 'operator': 'eq', 'value': kind}
        for cutoff in range(100, 2100, 100):
            for op, word in [('gt', 'above'), ('lt', 'below')]:
                yield (f'Show {noun} {word} {cutoff}', f'Which {kind.lower()} transactions are strictly {word} {cutoff}?',
                       dict(operation='filter', column='Amount', operator=op, value=cutoff, where=condition))
        for function, word in [('sum', 'total'), ('mean', 'average')]:
            common = dict(column='Amount', function=function, where=condition)
            yield (f'Show monthly {word} {noun}', f'For each calendar month give the {word} {noun}',
                   dict(operation='trend', group_by='Date', frequency='month', **common))
            for group in ['Category', 'Sub category']:
                yield (f'Show {word} {noun} by {group}', f'For each {group} calculate {word} {noun}',
                       dict(operation='aggregate', group_by=group, **common))
                for direction, label in [('desc', 'highest'), ('asc', 'lowest')]:
                    for limit in range(1, 6):
                        yield (f'Show {limit} {group} groups with {label} {word} {noun}',
                               f'By {word} {noun}, which {limit} {group} groups rank {label}?',
                               dict(operation='rank', group_by=group, direction=direction, limit=limit, **common))

rows = list(examples())
train = [(question, plan) for question, _, plan in rows]
test = []
for op, count in [('filter',4), ('aggregate',4), ('rank',4), ('trend',3)]:
    candidates = [(question, plan) for _, question, plan in rows if plan['operation'] == op]
    test.extend(random.Random(481).sample(candidates, count))
assert len({q for q,_ in test}) == 15
assert not {q for q,_ in train}.intersection(q for q,_ in test)
torch.manual_seed(42)
random.Random(42).shuffle(train)
prompt=lambda q:'Personal finance query to JSON plan. Columns='+json.dumps(F)+' Question='+q+' Plan='
assert torch.cuda.is_available()
name='Qwen/Qwen2.5-0.5B-Instruct'; tok=AutoTokenizer.from_pretrained(name)
base=AutoModelForCausalLM.from_pretrained(name,device_map={'':0},quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16))
model=get_peft_model(prepare_model_for_kbit_training(base),LoraConfig(task_type='CAUSAL_LM',r=32,lora_alpha=64,target_modules=['q_proj','k_proj','v_proj','o_proj']))
opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=2e-4)
for e in range(3):
 for q,p in train:
  a=tok.encode(prompt(q)); b=tok.encode(json.dumps(p,separators=(',',':')),add_special_tokens=False)+[tok.eos_token_id]; ids=torch.tensor([a+b],device='cuda')
  model(input_ids=ids,labels=torch.tensor([[-100]*len(a)+b],device='cuda')).loss.backward(); opt.step(); opt.zero_grad()
 print('epoch',e+1,'complete',flush=True)
out=Path('/kaggle/working/finance_candidate_v2'); model.save_pretrained(out); tok.save_pretrained(out)
res=[]; model.eval()
for q,p in test:
 x=tok(prompt(q),return_tensors='pt').to('cuda')
 with torch.inference_mode(): y=model.generate(**x,max_new_tokens=100,do_sample=False,pad_token_id=tok.eos_token_id)
 raw=tok.decode(y[0,x.input_ids.shape[1]:],skip_special_tokens=True)
 try: pred=json.loads(raw)
 except: pred=None
 res.append({'question':q,'expected':p,'prediction':pred,'correct':pred==p})
(out/'evaluation.json').write_text(json.dumps(res,indent=2)); print('FINANCE_RESULT',sum(x['correct'] for x in res),'/',len(res),flush=True)
