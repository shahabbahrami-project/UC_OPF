from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.ll.state_ll import StateLL
from utils.beam_search import beam_search
import numpy as np
import math
from scipy.special import erfinv


class LL(object):
    
    NAME = 'll'  # Low-level problem
    
    SNR_THR_DB = 5  # Hardcoded  SNR threshold for URLLC users in db
    SNR_THR = 10**(SNR_THR_DB/10)
    
    CHANNEL_USE = {
        0: 24.,
        1: 48.,
        2: 96.,
        }
    
    EPSILON = 1e-4
    
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
        
        chGain = dataset['chGain'] #(batch_size, n_loc)
        dataLoad = dataset['dataLoad'] #(batch_size, n_users)
        puncFlag = dataset['puncFlag'] #(batch_size, n_RBs)
        numerology = dataset['numerology'] #(batch_size)
        availRBNum = dataset['availRBNum'] #(batch_size)
        
        batch_size, n_loc = chGain.size()
        _, n_users = dataLoad.size()
        _, n_RBs = puncFlag.size()
        
        depotChGain = torch.zeros(batch_size, dtype=torch.float, device=chGain.device)
        chCond = torch.cat((depotChGain[:, None], chGain), -1)
        dataRate=torch.zeros_like(dataLoad)
        remainingDemands=dataLoad.clone()
        remainingUnusedRB= availRBNum - torch.count_nonzero(puncFlag,dim=-1)
        consumedPower = torch.zeros(batch_size, dtype=torch.float, device=chGain.device)
        puncRBsNum=torch.zeros(batch_size, dtype=torch.int64, device=chGain.device)
        ids=torch.arange(batch_size, dtype=torch.int64, device=chGain.device)
        for i in range(pi.size(1)-1):
            
            selected = pi[:,i]
            ids_withoutDepot = ids[selected >= 1]
            
            selectedUsers = (selected[ids_withoutDepot] - 1) // n_RBs # -1 because of depot
            selectedRBs = (selected[ids_withoutDepot] - 1) % n_RBs   # -1 because of depot
            
            ids_selectedPuncRBs = ids_withoutDepot[puncFlag[ids_withoutDepot, selectedRBs]==1]
            ids_selectedUnusedRBs = ids_withoutDepot[puncFlag[ids_withoutDepot, selectedRBs]==0]
            
            puncRBsNum[ids_selectedPuncRBs] +=1
            remainingUnusedRB[ids_selectedUnusedRBs] -= 1
            
            cur_chCond = chCond[ids, selected]
            
            allocatedPower = LL.SNR_THR/cur_chCond[ids_withoutDepot]
            #consumedPower[ids_withoutDepot] += allocatedPower
            consumedPower[ids_withoutDepot] += 10.0 * torch.log10(allocatedPower)
            
            chDispersion = 1. - (1. / ((1. + allocatedPower * cur_chCond[ids_withoutDepot])**2))
            chUse = torch.tensor(list(map(LL.CHANNEL_USE.get, numerology[ids_withoutDepot].tolist()))).to(chCond.device)    
            qFuncInv = math.sqrt(2) * erfinv(1- (2 * LL.EPSILON))
            
            
            dataRate[ids_withoutDepot,selectedUsers] += (1./math.log(2)) * (chUse * (torch.log(1 + allocatedPower * cur_chCond[ids_withoutDepot])) -
                                                                       torch.sqrt(chUse * chDispersion) * qFuncInv)
            
            remainingDemands[ids_withoutDepot,selectedUsers] -= dataRate[ids_withoutDepot,selectedUsers]
            remainingDemands[ids_withoutDepot,selectedUsers] = torch.max(remainingDemands[ids_withoutDepot,selectedUsers],torch.tensor([0]).to(chCond.device))
        
        
        return (
            consumedPower / 60.0
            + torch.count_nonzero((remainingDemands>0).float(),dim=-1) 
            + ((remainingUnusedRB > 0) & (puncRBsNum > 0)).float()
            ), None
        
        
        
    @staticmethod
    def make_dataset(*args, **kwargs):
        return LLDataset(*args, **kwargs)
    
    @staticmethod
    def make_state(*args, **kwargs):
        return StateLL.initialize(*args, **kwargs)
    
    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = LL.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)
    

def generate_instance(size, userTraffic):
    
    #numerology = torch.randint(0, 3, size=(1, )) # Numerology is selected uniformly random from the set {0, 1, 2}
    numerology = np.random.randint(0, 3) # Numerology is selected uniformly random from the set {0, 1, 2}
    
    max_n_RBs = 12#48
    assert size % max_n_RBs == 0
    n_users = int(size/max_n_RBs)
    
    trafficParam = 40. * torch.ones(n_users)  #400. * torch.ones(n_users) # 50 bytes per mini slot time
    dataLoad = torch.poisson(trafficParam)
    
    RB_step = 4
    max_wideRB = int(max_n_RBs/RB_step)
    
    n_eMBB_wideRBs = np.random.randint(0, max_wideRB+1)
    n_shared_wideRBs = np.random.randint(0, max_wideRB-n_eMBB_wideRBs+1)
    n_URLLC_wideRBs = max_wideRB-(n_eMBB_wideRBs+n_shared_wideRBs)
    
    assert n_eMBB_wideRBs + n_shared_wideRBs + n_URLLC_wideRBs == max_wideRB
    
    
    if numerology == 0:
        RB_step4Num = 4
    
    elif numerology == 1:
        RB_step4Num = 2
        
    else:
        RB_step4Num = 1
        
        
    #n_eMBB_RBs = n_eMBB_wideRBs * RB_step4Num # n_eMBB_RBs is not important here
    n_shared_RBs = n_shared_wideRBs * RB_step4Num
    n_URLLC_RBs = n_URLLC_wideRBs * RB_step4Num
    
    totURLLC_RBs = n_shared_RBs + n_URLLC_RBs
    #availRBNum = torch.tensor([totURLLC_RBs])
    availRBNum = totURLLC_RBs
    
    puncRate = 0.2 * np.random.uniform(0,1) + 0.8
    numOfPuncRBs = int(puncRate * n_shared_RBs)
    
    puncFlag = torch.zeros(max_n_RBs, dtype=torch.long)
    puncFlag[torch.randperm(n_shared_RBs)[:numOfPuncRBs]] = 1

    
    posTheta = 2 * np.pi * torch.rand(n_users,1)
    cellRadius = 500 # in meters
    posOffset = 50 # in meters
    posRadius = (cellRadius-posOffset) * torch.rand(n_users,1) + posOffset
    
    pos_x = posRadius * torch.cos(posTheta)
    pos_y = posRadius * torch.sin(posTheta)
    
    usersPos = torch.cat((pos_x, pos_y), 1)
    userDist = torch.linalg.norm(usersPos,dim=1)
    
    shadowingSTD = 10 # in db
    pathLoss_db = 128.1 + 37.6 * torch.log10(userDist/1000) + shadowingSTD * torch.randn(n_users)
    
    chGain_db = 20 * torch.log10(torch.abs(torch.randn(size, dtype=torch.cfloat)/np.sqrt(2))) - torch.repeat_interleave(pathLoss_db, max_n_RBs)
    
    noisePower_db = -110 # dbm
    noisePower = 10 ** ((noisePower_db-30)/10)
    chGain = torch.pow(10., chGain_db/10.) / noisePower
    
    dummyChIndices = torch.add(max_n_RBs * torch.arange(0,n_users).reshape((n_users,1)),torch.arange(availRBNum,max_n_RBs)).reshape((-1,))
    chGain[dummyChIndices] = 0.0
    
    
    return {
        'chGain': chGain,
        'dataLoad': dataLoad,
        'puncFlag': puncFlag,
        'numerology': torch.tensor(numerology),
        'availRBNum': torch.tensor(totURLLC_RBs)
    }


class LLDataset(Dataset):
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution='poisson'):
        super(LLDataset, self).__init__()
        
        assert distribution is not None, "Data distribution must be specified for URLLC Traffic"
        
        userTraffic = distribution
        
        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [
                    {
                        'chGain': torch.FloatTensor(chGain),      #(batch_size, n_loc)
                        'dataLoad': torch.FloatTensor(dataLoad),  #(batch_size, n_users)
                        'puncFlag': torch.FloatTensor(puncFlag),  #(batch_size, n_RBs)
                        'numerology': torch.tensor(numerology),   #(batch_size)
                        'availRBNum': torch.tensor(availRBNum)    #(batch_size)
                    }
                    for chGain, dataLoad, puncFlag, numerology, availRBNum in (data[offset:offset+num_samples])
                ]
                
        else:
            self.data = [
                generate_instance(size, userTraffic)
                for i in range(num_samples)
            ]
            
        self.size = len(self.data)
        
        
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


