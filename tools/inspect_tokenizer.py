import json,sys
p=sys.argv[1]
t=json.load(open(p,encoding='utf-8'))
print(t.keys())
print('model',t.get('model',{}).keys())
print('added',len(t.get('added_tokens',[])))
print('first vocab',list(t['model'].get('vocab',{}).items())[:5] if isinstance(t['model'].get('vocab'),dict) else type(t['model'].get('vocab')))
print('pre_tokenizer',t.get('pre_tokenizer'))
print('decoder',t.get('decoder'))
