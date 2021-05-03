from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.uc.state_uc import StateUC
from utils.beam_search import beam_search
import numpy as np
import cvxpy as cp
import math
from scipy.special import erfinv


from pypower.api import ppoption, runpf, runopf, opf, printpf
from pypower.makeYbus import makeYbus 
from pypower.ext2int import ext2int 
from pypower.loadcase import loadcase 
from pypower.api import case14 as case
import numpy as np
import json
from OPF_Fast import OPFAC


class UC(object):
    
    NAME = 'uc'  # unit commitment problem
    
    @staticmethod
    def get_costs(dataset, pi):

        if pi.size(-1) == 1:  # In case all tours directly return to depot, prevent further problems
            assert (pi == 0).all(), "If all length 1 tours, they should be zero"
            # Return
            return torch.zeros(pi.size(0), dtype=torch.float, device=pi.device), None                  

         # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]
        # Make sure each node visited once at most (except for depot)
        assert ((sorted_pi[:, 1:] == 0) | (sorted_pi[:, 1:] > sorted_pi[:, :-1])).all(), "Duplicates"


        ck0 = dataset['ck0'].cpu().numpy() #(batch_size, n_loc)
        ck1 = dataset['ck1'].cpu().numpy() #(batch_size, n_users)
        ck2 = dataset['ck2'].cpu().numpy()  #(batch_size, n_RBs)
        tu = dataset['tu'].cpu().numpy() #(batch_size)
        td = dataset['td'].cpu().numpy() #(batch_size)
        pMin = dataset['pMin'].cpu().numpy() #(batch_size)
        pMax = dataset['pMax'].cpu().numpy() #(batch_size)
        load = dataset['load'].cpu().numpy() #(batch_size)
        totalLoad = dataset['totalLoad'].cpu().numpy() #(batch_size)
        qload = dataset['qload'].cpu().numpy() #(batch_size)
        totalqLoad = dataset['totalqLoad'].cpu().numpy() #(batch_size)
        batch_size, ngen = ck0.size()
        u=torch.zeros(batch_size, ngen, dtype=torch.float,device=ck0.devic)
        zeroTensor=torch.zeros(batch_size, ngen, dtype=torch.float,device=ck0.devic)
        u=torch.where(pi<=ngen, 1-zeroMatrix, zeroMatrix)
        weight=10000000
        cost=torch.zeros(batch_size, 1, dtype=torch.float,device=ck0.devic)
        ppc = loadcase(case())
        for i in range(pi.size(0)):
           cost[i,1]=OPFAC(ppc, pload[i,:], qload[i,:], u[i,:], weight)
        return (
            cost
        ), None
    
    
    @staticmethod
    def make_dataset(*args, **kwargs):
        return UCDataset(*args, **kwargs)
    
    @staticmethod
    def make_state(*args, **kwargs):
        return StateUC.initialize(*args, **kwargs)
    
    
    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = UC.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


def generate_instance(size, userTraffic):
    
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
    qqload = torch.tensor(ppc['bus'][:,3])
    qloadAllHours = qqload.repeat(T, 1)
    qload = torch.normal(mean=qloadAllHours, std=0.05*qloadAllHours)/baseMVA  # 24 rows, each is the load profile of the grid         
    totalqLoad = torch.sum(qload, dim=1)   



    ck2 = torch.tensor(ppc['gencost'][:,4])
    ck1 = torch.tensor(ppc['gencost'][:,5])
    ck0 = torch.tensor(ppc['gencost'][:,6])
    
    tu = torch.randint(0, 2, (n_gen,))
    td = torch.randint(0, 2, (n_gen,))
    
    genMatrix=np.zeros((n_bus,ngenInfo))
    for i in range(n_gen):
        genMatrix[int(ppc['gen'][i][0])-1,:]=ppc['gen'][i]

    Pmax=torch.tensor(ppc['gen'][:,8])/baseMVA
    Pmin=torch.tensor(ppc['gen'][:,9])/baseMVA

    return {
            'ck0': ck0,
            'ck1': ck1,
            'ck2': ck2,
            'tu': tu,
            'td':  td,
            'pMin': pMin,
            'pMax': pMax, 
            'load': load,
            'totalLoad': totalLoad,
            'qload': qload,
            'totalqLoad': totalqLoad
    }   




class UCDataset(Dataset):
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution='Normal'):
        super(UCDataset, self).__init__()
        
        assert distribution is not None, "Data distribution must be specified for URLLC Traffic"
        
        loadDemand = distribution
        
        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [
                    {
                        'ck0': torch.FloatTensor(ck0),      #(batch_size, n_gen) 
                        'ck1': torch.FloatTensor(ck1), #(batch_size, n_gen)
                        'ck2': torch.FloatTensor(ck2),  #(batch_size, n_gen)
                        'tu': torch.tensor(pMin),   #(batch_size, n_gen)
                        'td': torch.tensor(pMin),   #(batch_size, n_gen)  
                        'pMin': torch.tensor(pMin),   #(batch_size, n_gen)
                        'pMax': torch.tensor(pMax),   #(batch_size, n_gen)                         
                        'load': torch.tensor(load)   #(batch_size, T, n_bus)
                        'totalLoad': torch.tensor(totalLoad)   #(batch_size, T)
                        'qload': torch.tensor(qload)   #(batch_size, T, n_bus)
                        'totalqLoad': torch.tensor(totalqLoad)   #(batch_size, T)
                    }
                    for ck0, ck1, ck2, tu, td, pMin, pMax, load, totalload, qload, totalqLoad in (data[offset:offset+num_samples])
                ]
                
        else:
            self.data = [
                generate_instance(size)
                for i in range(num_samples)
            ]
            
        self.size = len(self.data)
        
        
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
    
