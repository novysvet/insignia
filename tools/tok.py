#!/usr/bin/env python3
import argparse,json,pathlib,struct
from tokenizers import Tokenizer

def main():
 p=argparse.ArgumentParser();p.add_argument('model_dir',type=pathlib.Path);p.add_argument('text');p.add_argument('--chat',action='store_true');p.add_argument('--decode',type=int,nargs='*');a=p.parse_args()
 tok=Tokenizer.from_file(str(a.model_dir/'tokenizer.json'))
 if a.decode is not None:
  print(tok.decode(a.decode,skip_special_tokens=False));return
 text=a.text
 if a.chat:
  tpl=(a.model_dir/'chat_template.jinja').read_text(encoding='utf-8')
  try:
   from jinja2 import Environment
  except ImportError: raise SystemExit('pip install jinja2')
  text=Environment().from_string(tpl).render(messages=[{'role':'user','content':text}],add_generation_prompt=True,enable_thinking=False,tools=None)
 ids=tok.encode(text,add_special_tokens=False).ids
 print(json.dumps({'text':text,'ids':ids},ensure_ascii=False))
if __name__=='__main__':main()
