from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.ml.state_ml import StateML
from utils.beam_search import beam_search
import numpy as np
import cvxpy as cp
import math
from scipy.special import erfinv



class ML(object):
    
    NAME = 'ml'  # medium-level problem
    
    PRB_BW = {
        0: 180.,
        1: 360.,
        2: 720.,
        }
    
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
        
        
        chGain = dataset['chGain'].cpu().numpy() #(batch_size, n_loc)
        minRateReq = dataset['minRateReq'].cpu().numpy() #(batch_size, n_users)
        sharedFlag = dataset['sharedFlag'].cpu().numpy()  #(batch_size, n_RBs)
        numerology = dataset['numerology'].cpu().numpy() #(batch_size)
        availRBNum = dataset['availRBNum'].cpu().numpy() #(batch_size)
        availPower = dataset['availPower'].cpu().numpy() #(batch_size)
        
        batch_size, n_loc = chGain.shape
        _, n_users = minRateReq.shape
        _, n_RBs = sharedFlag.shape
        
        ids=torch.arange(pi.size(1), dtype=torch.int64)
        penaltyFactor = 1000.
        scaleFactor = 1000.
        
        penalizedAggregateThroughput = torch.zeros(batch_size, dtype=torch.float, device = pi.device)
        
        for i in range(pi.size(0)):
            
            selected = pi[i,:]
            ids_withoutDepot = ids[selected >= 1]
            
            selectedUsers = ((selected[ids_withoutDepot] - 1) // n_RBs).cpu().numpy() # -1 because of depot
            selectedRBs = ((selected[ids_withoutDepot] - 1) % n_RBs).cpu().numpy()  # -1 because of depot
            
            U = n_users
            K = availRBNum[i]
            
            if K==0:
                penalizedAggregateThroughput[i] = 0.
                continue
                
            alpha_val = np.zeros((U,K), dtype=int)
            alpha_val[selectedUsers, selectedRBs] = 1
            
            P = cp.Variable(shape=(U,K))
            Delta = cp.Variable(shape=U)
            g_val = chGain[i,:]
            gain = g_val[g_val != 0].reshape((U, K))
            
            b_embb = ML.PRB_BW[numerology[i]]/scaleFactor
        
            # This function will be used as the objective so must be DCP;
            # i.e. elementwise multiplication must occur inside kl_div,
            # not outside otherwise the solver does not know if it is DCP...
            R = cp.multiply((b_embb/np.log(2)), cp.multiply(alpha_val, cp.log1p(cp.multiply(P,gain))))
        
            objective = cp.Minimize(-cp.sum(R)+ cp.multiply(penaltyFactor, cp.sum(cp.power(Delta, 2))))
            constraints = [cp.sum(cp.multiply(alpha_val, P)) <= availPower[i],
                           (Delta + cp.sum(R, axis=1))>= (minRateReq[i,:]/scaleFactor),
                           P>=0.0,
                           Delta>=0.0]

            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.MOSEK)
            
            penalizedAggregateThroughput[i] = prob.value
                  
        
        return (
            penalizedAggregateThroughput
            ), None
    
    @staticmethod
    def make_dataset(*args, **kwargs):
        return MLDataset(*args, **kwargs)
    
    @staticmethod
    def make_state(*args, **kwargs):
        return StateML.initialize(*args, **kwargs)
    
    
    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = ML.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)
    
def generate_instance(size, userTraffic):
    
    numerology = np.random.randint(0, 3) # Numerology is selected uniformly random from the set {0, 1, 2}
    
    max_n_RBs = 12#48
    assert size % max_n_RBs == 0
    n_users = int(size/max_n_RBs)
    
    minRateReq = 6000. * torch.ones(n_users) # 6000 Kb/s 
    
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
        
        
    n_eMBB_RBs = n_eMBB_wideRBs * RB_step4Num 
    n_shared_RBs = n_shared_wideRBs * RB_step4Num
    #n_URLLC_RBs = n_URLLC_wideRBs * RB_step4Num # n_URLLC_RBs is not important here
    
    toteMBB_RBs = n_eMBB_RBs + n_shared_RBs
    #availRBNum = torch.tensor([totURLLC_RBs])
    availRBNum = toteMBB_RBs
    
    sharedFlag = torch.zeros(max_n_RBs, dtype=torch.long)
    sharedFlag[0:n_shared_RBs] = 1
    
    
    maxTxPower = 1 # in watt
    powFactorSet = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    selectedPowFactorIdx = np.random.randint(0, len(powFactorSet))
    availPower = powFactorSet[selectedPowFactorIdx] * maxTxPower
    
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
        'minRateReq': minRateReq,
        'sharedFlag': sharedFlag,
        'numerology': torch.tensor(numerology),
        'availRBNum': torch.tensor(availRBNum),
        'availPower': torch.tensor(availPower)
    }    
    
    
    
    
class MLDataset(Dataset):
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution='poisson'):
        super(MLDataset, self).__init__()
        
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
                        'minRateReq': torch.FloatTensor(minRateReq), #(batch_size, n_users)
                        'sharedFlag': torch.FloatTensor(sharedFlag),  #(batch_size, n_RBs)
                        'numerology': torch.tensor(numerology),   #(batch_size)
                        'availRBNum': torch.tensor(availRBNum),   #(batch_size)
                        'availPower': torch.tensor(availPower)    #(batch_size)
                    }
                    for chGain, minRateReq, sharedFlag, numerology, availRBNum, availPower in (data[offset:offset+num_samples])
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
    