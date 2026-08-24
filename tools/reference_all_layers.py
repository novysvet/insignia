import json,struct,pathlib,sys,numpy as np
p=pathlib.Path(sys.argv[1]);f=p.open('rb');n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n));start=8+n
def get(k):
 v=h[k];f.seek(start+v['data_offsets'][0]);raw=f.read(v['data_offsets'][1]-v['data_offsets'][0]);dt={'U32':'<u4','U8':'u1','BF16':'<u2','F32':'<f4'}[v['dtype']];a=np.frombuffer(raw,dtype=dt).reshape(v['shape']);return (a.astype(np.uint32)<<16).view(np.float32) if v['dtype']=='BF16' else a
def dq(base):
 w=get(base+'.weight');s=get(base+'.scales');rows=w.shape[0];cols=w.shape[1]*8;u=w.reshape(rows,-1);q=((u[:,:,None]>>(np.arange(8,dtype=np.uint32)*4))&15).reshape(rows,cols);lut=np.array([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],np.float32);scale=(s.astype(np.uint32)<<23).view(np.float32);return lut[q]*np.repeat(scale,32,axis=1)
def rms(x,w):return x/np.sqrt(np.mean(x*x)+1e-6)*w
def mlp(x,p):
 n=rms(x,get(p+'.post_attention_layernorm.weight'));g=dq(p+'.mlp.gate_proj')@n;u=dq(p+'.mlp.up_proj')@n;return x+dq(p+'.mlp.down_proj')@(g/(1+np.exp(-g))*u)
def delta(x,l):
 p=f'language_model.model.layers.{l}';nrm=rms(x,get(p+'.input_layernorm.weight'));a=p+'.linear_attn';qkv=dq(a+'.in_proj_qkv')@nrm;z=dq(a+'.in_proj_z')@nrm;aa=dq(a+'.in_proj_a')@nrm;bb=dq(a+'.in_proj_b')@nrm;cw=get(a+'.conv1d.weight').reshape(8192,4);qkv=qkv*cw[:,3];qkv=qkv/(1+np.exp(-qkv));q,k,v=np.split(qkv,[2048,4096]);q=q.reshape(16,128);k=k.reshape(16,128);v=v.reshape(32,128);q=q/(np.sqrt(np.mean(q*q,1,keepdims=True)+1e-6)*128);k=k/(np.sqrt(np.mean(k*k,1,keepdims=True)+1e-6)*np.sqrt(128));q=np.repeat(q,2,0);k=np.repeat(k,2,0);beta=1/(1+np.exp(-bb));di=l-l//4;state=states[di];state*=np.exp(-np.exp(get(a+'.A_log'))*np.logaddexp(0,aa+get(a+'.dt_bias')))[:,None,None];mem=np.einsum('hvk,hk->hv',state,k);dd=(v-mem)*beta[:,None];state+=dd[:,:,None]*k[:,None,:];out=np.einsum('hvk,hk->hv',state,q);nw=get(a+'.norm.weight');out=out/np.sqrt(np.mean(out*out,1,keepdims=True)+1e-6)*nw[None,:]*(z.reshape(32,128)/(1+np.exp(-z.reshape(32,128))));return mlp(x+dq(a+'.out_proj')@out.reshape(-1),p)
def attn(x,l):
 p=f'language_model.model.layers.{l}';a=p+'.self_attn';nrm=rms(x,get(p+'.input_layernorm.weight'));raw=(dq(a+'.q_proj')@nrm).reshape(16,512);q=raw[:,:256];gate=raw[:,256:];k=(dq(a+'.k_proj')@nrm).reshape(4,256);v=(dq(a+'.v_proj')@nrm).reshape(4,256);q=q/np.sqrt(np.mean(q*q,1,keepdims=True)+1e-6)*get(a+'.q_norm.weight');k=k/np.sqrt(np.mean(k*k,1,keepdims=True)+1e-6)*get(a+'.k_norm.weight');out=np.repeat(v,4,axis=0);out*=1/(1+np.exp(-gate));return mlp(x+dq(a+'.o_proj')@out.reshape(-1),p)
x=dq('language_model.model.embed_tokens')[42];states=[np.zeros((32,128,128),np.float32) for _ in range(24)];native=np.fromfile(sys.argv[2],np.float32).reshape(32,4096)
for l in range(32):
 x=attn(x,l) if l%4==3 else delta(x,l);d=native[l];print(l,'A' if l%4==3 else 'D','max',float(abs(x-d).max()),'mean',float(abs(x-d).mean()),'cos',float(x@d/np.linalg.norm(x)/np.linalg.norm(d)))
