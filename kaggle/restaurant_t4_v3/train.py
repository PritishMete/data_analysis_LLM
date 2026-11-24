import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'bitsandbytes>=0.46.1'])
import json, random, zipfile
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

FIELDS=['id','name','city','rating','rating_count','cuisine','cost','link']
examples=[]
for city in ['Bangalore','Kolkata','Mumbai','Delhi','Chennai']:
    examples += [(f'Find restaurants in {city}.',{'operation':'filter','column':'city','operator':'eq','value':city}), (f'Show the top 5 rated restaurants in {city}.',{'operation':'rank','column':'rating','direction':'desc','limit':5,'where':{'column':'city','operator':'eq','value':city}})]
for n in [3,5,10]:
    examples += [(f'Show top {n} rated restaurants.',{'operation':'rank','column':'rating','direction':'desc','limit':n}), (f'List the {n} most expensive restaurants.',{'operation':'rank','column':'cost','direction':'desc','limit':n})]
for rating in [3.5,4.0,4.5]:
    examples += [(f'Show restaurants rated above {rating}.',{'operation':'filter','column':'rating','operator':'gt','value':rating})]
for cuisine in ['Biryani','Chinese','North Indian','South Indian','Italian']:
    examples += [(f'Find restaurants serving {cuisine}.',{'operation':'filter','column':'cuisine','operator':'eq','value':cuisine})]
for group,col,fn in [('city','rating','mean'),('city','cost','mean'),('cuisine','rating','mean'),('cuisine','cost','mean')]:
    examples += [(f'Show the average {col} by {group}.',{'operation':'aggregate','group_by':group,'column':col,'function':fn})]
random.Random(42).shuffle(examples)
test=examples[:max(20,len(examples)//5)]; train=examples[max(20,len(examples)//5):]
for city in ['Pune','Hyderabad','Jaipur','Lucknow','Surat','Patna']:
    city_filter = dict(column='city', operator='eq', value=city)
    train.append((f'List dining places located in {city}', dict(operation='filter', **city_filter)))
    for n in [2,4,6,8]:
        train.append((f'Which {n} restaurants in {city} have the highest ratings?',
                      dict(operation='rank', column='rating', direction='desc', limit=n, where=city_filter)))
for n in range(1,17):
    for column, phrase in [('cost','most expensive'),('rating','highest rated')]:
        train.append((f'Give me the {n} {phrase} dining places',
                      dict(operation='rank', column=column, direction='desc', limit=n)))
for threshold in [2.5,3.1,3.4,3.7,3.9,4.1,4.2,4.3,4.6,4.8]:
    train.append((f'Which places have a rating greater than {threshold}?',
                  dict(operation='filter', column='rating', operator='gt', value=threshold)))
for cuisine in ['Thai','Mexican','Korean','Japanese','French','Lebanese']:
    train.append((f'List dining places offering {cuisine} food',
                  dict(operation='filter', column='cuisine', operator='eq', value=cuisine)))
for group in ['city','cuisine']:
    for column in ['cost','rating','rating_count']:
        for question in [f'Calculate mean {column} per {group}', f'For each {group} give average {column}']:
            train.append((question, dict(operation='aggregate',group_by=group,column=column,function='mean')))
assert len(test) == 20 and not ({q.casefold() for q,_ in train} & {q.casefold() for q,_ in test})
random.Random(121).shuffle(train)
torch.manual_seed(121)
print('training examples',len(train),'evaluation examples',len(test),flush=True)
prompt=lambda q:'Restaurant query to JSON plan. Columns='+json.dumps(FIELDS)+' Question='+q+' Plan='
assert torch.cuda.is_available()
tok=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
base=AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct',device_map={'':0},quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16))
model=get_peft_model(prepare_model_for_kbit_training(base),LoraConfig(task_type='CAUSAL_LM',r=32,lora_alpha=64,target_modules=['q_proj','k_proj','v_proj','o_proj']))
opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=2e-4)
for epoch in range(3):
    model.train()
    for q,plan in train:
        a=tok.encode(prompt(q)); b=tok.encode(json.dumps(plan,separators=(',',':')),add_special_tokens=False)+[tok.eos_token_id]
        ids=torch.tensor([a+b],device='cuda'); loss=model(input_ids=ids,labels=torch.tensor([[-100]*len(a)+b],device='cuda')).loss
        loss.backward(); opt.step(); opt.zero_grad()
    print('epoch',epoch+1,'complete',flush=True)
out=Path('/kaggle/working/restaurant_candidate_v3'); out.mkdir(exist_ok=True); model.save_pretrained(out); tok.save_pretrained(out)
model.eval(); results=[]
for q,expected in test:
    x=tok(prompt(q),return_tensors='pt').to('cuda')
    with torch.inference_mode(): ids=model.generate(**x,max_new_tokens=120,do_sample=False,pad_token_id=tok.eos_token_id)
    raw=tok.decode(ids[0,x.input_ids.shape[1]:],skip_special_tokens=True)
    try: pred=json.loads(raw)
    except ValueError: pred=None
    results.append({'question':q,'expected':expected,'prediction':pred,'correct':pred==expected})
(out/'evaluation.json').write_text(json.dumps(results,indent=2)); correct=sum(r['correct'] for r in results); print('RESTAURANT_V3_RESULT',correct,'/',len(results),flush=True)
with zipfile.ZipFile('/kaggle/working/restaurant_candidate_v3_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in out.rglob('*'):
        if p.is_file(): z.write(p,p.relative_to(out))
print('/kaggle/working/restaurant_candidate_v3_bundle.zip',flush=True)
