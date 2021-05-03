from pypower.loadcase import loadcase 
from pypower.api import case14 as case
import torch
import numpy as np



# ppc = loadcase(case()) 
# ck2=ppc['gencost'][:,4]
# ck1=ppc['gencost'][:,5]
# ck0=ppc['gencost'][:,6]
# n_gen = ppc['gen'].shape[0]  
# tu = 1*torch.ones(n_gen)
# n_bus=ppc['bus'].shape[0]
# ngenInfo=ppc['gen'].shape[1]
# genMatrix=np.zeros((n_bus,ngenInfo))
# for i in range(n_gen):
#     genMatrix[int(ppc['gen'][i][0])-1,:]=ppc['gen'][i]

# Pmax=torch.tensor(ppc['gen'][:,8])/baseMVA
# Pmin=torch.tensor(ppc['gen'][:,9])/baseMVA

def generate_instance():
    
    ppc = loadcase(case())
    baseMVA=ppc['baseMVA']
    n_gen = ppc['gen'].shape[0] 
    n_bus=ppc['bus'].shape[0]
    ngenInfo=ppc['gen'].shape[1]   
    T = 24

    pload = torch.tensor(ppc['bus'][:,2])
    ploadAllHours = pload.repeat(T, 1)
    load = torch.normal(mean=ploadAllHours, std=0.05*ploadAllHours)/baseMVA  # 24 rows, each is the load profile of the grid         
    totalLoad = torch.sum(load, dim=1)        # 24 numbers each is the total load of that time slot
    
    ck2 = torch.tensor(ppc['gencost'][:,4])
    ck1 = torch.tensor(ppc['gencost'][:,5])
    ck0 = torch.tensor(ppc['gencost'][:,6])
    
    tu = torch.randint(0, 2, (n_gen,))
    td = torch.randint(0, 2, (n_gen,))
    
    genMatrix=np.zeros((n_bus,ngenInfo))
    for i in range(n_gen):
        genMatrix[int(ppc['gen'][i][0])-1,:]=ppc['gen'][i]

    pMax=torch.tensor(ppc['gen'][:,8])/baseMVA
    pMin=torch.tensor(ppc['gen'][:,9])/baseMVA

    return {
            'ck0': ck0,
            'ck1': ck1,
            'ck2': ck2,
            'tu': tu,
            'td':  td,
            'pMin': pMin,
            'pMax': pMax, 
            'load': load,
            'totalLoad': totalLoad
    }   

data = [
    generate_instance()
    for i in range(2)
]