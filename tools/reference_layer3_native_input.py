exec(open('tools/reference_all_layers.py').read().split("x=dq('language_model.model.embed_tokens')[42]")[0])
native=np.fromfile(sys.argv[2],np.float32).reshape(32,4096);x=native[2].copy();y=attn(x,3);d=native[3];print('native-input layer3','max',abs(y-d).max(),'mean',abs(y-d).mean(),'cos',y@d/np.linalg.norm(y)/np.linalg.norm(d),y[:8],d[:8])
