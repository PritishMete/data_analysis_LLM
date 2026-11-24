import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'bitsandbytes>=0.46.1'])
import json, random, zipfile
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

FIELDS = 'product_name brand_name rating rating_count marked_price discounted_price sizes product_link'.split()
train, test = [], []
families = [
 ('product_name','rating','desc',None,'highest rated products','products with the highest ratings'),
 ('product_name','rating_count','desc',None,'most reviewed products','products with the largest review counts'),
 ('product_name','discounted_price','asc',None,'cheapest products on sale','items with the lowest discounted prices'),
 ('brand_name','rating','desc','mean','brands by average rating','brands with the highest mean product ratings'),
 ('brand_name','product_name','desc','count','brands by product count','brands containing the most items'),
]
for group, metric, direction, function, phrase, paraphrase in families:
    for n in range(1, 19):
        expected = dict(operation='rank', group_by=group, column=metric, direction=direction, limit=n)
        if function:
            expected['function'] = function
        if n in (4, 7, 11):
            test.append((f'Which {n} {paraphrase} lead the list?', expected))
        else:
            for template in ('Show the top {n} {phrase}', 'List {n} {phrase}', 'Return the {n} {phrase}'):
                train.append((template.format(n=n, phrase=phrase), expected))
assert len(test)==15 and not ({q.casefold() for q,_ in train} & {q.casefold() for q,_ in test})
for n in range(1, 21):
    low = dict(operation='rank', group_by='product_name', column='discounted_price', direction='asc', limit=n)
    train.extend([
        (f'Find the {n} cheapest products by sale price', low),
        (f'List {n} items ordered from lowest discounted price', low),
        (f'Which {n} products have the smallest sale prices?', low),
    ])
# Reinforce every valid ranking family with alternate wording while keeping
# the held-out test wording distinct.
for group, metric, direction, function in [
    ('product_name','rating','desc',None),
    ('product_name','rating_count','desc',None),
    ('product_name','discounted_price','asc',None),
    ('brand_name','rating','desc','mean'),
    ('brand_name','product_name','desc','count'),
]:
    for n in range(1, 19):
        p={'operation':'rank','group_by':group,'column':metric,'direction':direction,'limit':n}
        if function: p['function']=function
        train.extend([(f'Rank exactly {n} records using {metric}',p),
                      (f'Give me {n} results sorted by {metric}',p)])
# Target the previously failing semantic family with varied, valid phrasing.
for n in range(1, 19):
    p={'operation':'rank','group_by':'brand_name','column':'rating','direction':'desc','limit':n,'function':'mean'}
    train.extend([
        (f'Rank {n} brands by their average rating',p),
        (f'Find {n} brands with the best mean ratings',p),
        (f'Show {n} brands ordered by average product rating',p),
        (f'List the top {n} brand averages for rating',p),
    ])
random.Random(42).shuffle(train)
torch.manual_seed(42)
prompt=lambda q:'Myntra query to JSON plan. Columns='+json.dumps(FIELDS)+' Question='+q+' Plan='
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
out=Path('/kaggle/working/myntra_candidate_v2'); out.mkdir(exist_ok=True); model.save_pretrained(out); tok.save_pretrained(out)
model.eval(); results=[]
for q,expected in test:
    x=tok(prompt(q),return_tensors='pt').to('cuda')
    with torch.inference_mode(): ids=model.generate(**x,max_new_tokens=120,do_sample=False,pad_token_id=tok.eos_token_id)
    raw=tok.decode(ids[0,x.input_ids.shape[1]:],skip_special_tokens=True)
    try: pred=json.loads(raw)
    except ValueError: pred=None
    results.append({'question':q,'expected':expected,'prediction':pred,'correct':pred==expected})
(out/'evaluation.json').write_text(json.dumps(results,indent=2)); correct=sum(r['correct'] for r in results); print('MYNTRA_V2_RESULT',correct,'/',len(results),flush=True)
with zipfile.ZipFile('/kaggle/working/myntra_candidate_v2_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in out.rglob('*'):
        if p.is_file(): z.write(p,p.relative_to(out))
print('/kaggle/working/myntra_candidate_v2_bundle.zip',flush=True)
