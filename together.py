import numpy as np
info=np.loadtxt('sun_17_eq',dtype=str)
n_source=len(info)
print(n_source)
import os
from multiprocessing import Pool

def run_python(i):
    print('Solving sun_17_%d'%(i))
    info_i=info[np.where(info[:,0]==str(i))]
    os.system('python HINSA_identify.py %d'%(i))

if __name__=='__main__':
    with Pool(10) as p:
        p.map(run_python,range(1,n_source))
