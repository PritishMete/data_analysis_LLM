import subprocess,sys
subprocess.check_call([sys.executable,'-m','pip','install','-q','-U','bitsandbytes>=0.46.1'])
import json,random
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
F=['Date','Product Category','Total Amount','Quantity','Gender','Price per Unit']; train=[]; test=[]
for g in ['Product Category','Gender','Date']:
 for c in ['Total Amount','Quantity']:
  for fn,word in [('sum','total'),('mean','average')]:
   p={'operation':'aggregate','group_by':g,'column':c,'function':fn}
   if len(test)<8:test.append((f'For every {g}, what is the {word} {c}?',p))
   train.extend([(f'Show {word} {c} by {g}',p),(f'Calculate {word} {c} for each {g}',p)])
for c in ['Total Amount','Quantity','Price per Unit']:
 for op,word in [('gt','above'),('lt','below')]:
  for n in [25,50,100,500,1000]: train.append((f'Show transactions with {c} {word} {n}',{'operation':'filter','column':c,'operator':op,'value':n}))
assert not ({q.casefold() for q,_ in train}&{q.casefold() for q,_ in test}); random.Random(42).shuffle(train)
prompt=lambda q:'Retail query to JSON plan. Columns='+json.dumps(F)+' Question='+q+' Plan='
assert torch.cuda.is_available(); name='Qwen/Qwen2.5-0.5B-Instruct'; tok=AutoTokenizer.from_pretrained(name)
base=AutoModelForCausalLM.from_pretrained(name,device_map={'':0},quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16))
model=get_peft_model(prepare_model_for_kbit_training(base),LoraConfig(task_type='CAUSAL_LM',r=32,lora_alpha=64,target_modules=['q_proj','k_proj','v_proj','o_proj'])); opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=2e-4)
for e in range(3):
 for q,p in train:
  a=tok.encode(prompt(q)); b=tok.encode(json.dumps(p,separators=(',',':')),add_special_tokens=False)+[tok.eos_token_id]; ids=torch.tensor([a+b],device='cuda'); model(input_ids=ids,labels=torch.tensor([[-100]*len(a)+b],device='cuda')).loss.backward(); opt.step(); opt.zero_grad()
 print('epoch',e+1,'complete',flush=True)
out=Path('/kaggle/working/retail_candidate_v2'); model.save_pretrained(out); tok.save_pretrained(out); model.eval(); res=[]
for q,p in test:
 x=tok(prompt(q),return_tensors='pt').to('cuda')
 with torch.inference_mode(): y=model.generate(**x,max_new_tokens=100,do_sample=False,pad_token_id=tok.eos_token_id)
 raw=tok.decode(y[0,x.input_ids.shape[1]:],skip_special_tokens=True)
 try: pred=json.loads(raw)
 except: pred=None
 res.append({'question':q,'expected':p,'prediction':pred,'correct':pred==p})
(out/'evaluation.json').write_text(json.dumps(res,indent=2)); print('RETAIL_RESULT',sum(x['correct'] for x in res),'/',len(res),flush=True)
