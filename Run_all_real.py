# run all experiments
from visualization import *
import numpy as np
import random
import subprocess

seed = 1
np.random.seed(seed=seed)#
random.seed(seed)
# print(random.random())

for sigma in [0.05, 0.1, 5.0]:
    for sigma_e in [0.1, 1.0, 2.0]:
        print('sigma: '+str(sigma)+', sigma_e: '+str(sigma_e))
        subprocess.call(["python", "RunExp_Real.py", '-T', '2500', '-n', '1000', '--sigma', str(sigma), '--warmup', '0.1', '--ind_S', '100', '--sigma_e', str(sigma_e), '-p', '0.2', '--save', '--seed', str(seed), '--save_name', 'sigma_'+str(sigma)+'_sigma_e_'+str(sigma_e)])
