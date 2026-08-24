import numpy as np,sys
all=np.fromfile(sys.argv[1],np.float32).reshape(32,4096)[3];one=np.fromfile(sys.argv[2],np.float32);print(abs(all-one).max(),abs(all-one).mean(),all@one/np.linalg.norm(all)/np.linalg.norm(one),all[:5],one[:5])
