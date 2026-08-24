exec(open('tools/reference_all_layers.py').read().split("x=dq('language_model.model.embed_tokens')[42]")[0])
states=[np.zeros((32,128,128),np.float32) for _ in range(24)]
x=dq('language_model.model.embed_tokens')[42]
for l in range(3):x=delta(x,l)
p='language_model.model.layers.3';a=p+'.self_attn';nrm=rms(x,get(p+'.input_layernorm.weight'));raw=(dq(a+'.q_proj')@nrm).reshape(16,512);q=raw[:,:256].copy();gate=raw[:,256:].copy();k=(dq(a+'.k_proj')@nrm).reshape(4,256);v=(dq(a+'.v_proj')@nrm).reshape(4,256);qn=q/np.sqrt(np.mean(q*q,1,keepdims=True)+1e-6)*get(a+'.q_norm.weight');kn=k/np.sqrt(np.mean(k*k,1,keepdims=True)+1e-6)*get(a+'.k_norm.weight');core=np.repeat(v,4,0);gated=core*(1/(1+np.exp(-gate)));op=dq(a+'.o_proj')@gated.reshape(-1);res=x+op;n2=rms(res,get(p+'.post_attention_layernorm.weight'));gg=dq(p+'.mlp.gate_proj')@n2;uu=dq(p+'.mlp.up_proj')@n2;ml=dq(p+'.mlp.down_proj')@(gg/(1+np.exp(-gg))*uu);final=res+ml
ref=[x,nrm,q.reshape(-1),gate.reshape(-1),k.reshape(-1),v.reshape(-1),qn.reshape(-1),1/(1+np.exp(-gate.reshape(-1))),kn.reshape(-1),core.reshape(-1),gated.reshape(-1),op,res,n2,gg,uu,ml,final]
native=np.fromfile(sys.argv[2],np.float32);off=0
for i,r in enumerate(ref):
 d=native[off:off+r.size];off+=r.size;print(i,r.shape,'max',abs(r-d).max(),'mean',abs(r-d).mean(),'cos',r@d/np.linalg.norm(r)/np.linalg.norm(d),'r',r[:3],'d',d[:3])
