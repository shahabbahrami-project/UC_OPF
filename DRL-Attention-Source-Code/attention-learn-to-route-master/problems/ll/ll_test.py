import sys, os
sys.path.append(os.path.abspath(os.path.join('../..')))
import json
from tqdm import tqdm
import torch
import numpy as np
from utils import load_problem
from torch.utils.data import DataLoader
import math
from scipy.special import erfinv
import pickle


def __DecimalToAnyBaseArrayRecur__(array, decimal, base):
    array.append(decimal % base)
    div = decimal // base
    if(div == 0):
        return;
    __DecimalToAnyBaseArrayRecur__(array, div, base)

def DecimalToAnyBaseArray(decimal, base):
    array = []
    __DecimalToAnyBaseArrayRecur__(array, decimal, base)
    return array[::-1]

file_path = os.path.abspath(os.path.join('../../outputs/ll_24/run_20210301T185411/args.json'))

print(file_path)

with open(file_path, 'r') as f:
    opts = json.load(f)
    

problem = load_problem(opts['problem'])

SNR_THR_DB = 5  # Hardcoded  SNR threshold for URLLC users in db
SNR_THR = 10**(SNR_THR_DB/10)
 
CHANNEL_USE = {
    0: 24.,
    1: 48.,
    2: 96.,
    }
 
EPSILON = 1e-4

num_test_samples = 1000

#test_dataset = problem.make_dataset(
#        size=opts['graph_size'], num_samples=num_test_samples, filename=None, distribution=opts['data_distribution'])

with open(os.path.join('../../', opts['save_dir'], "valDataset"), 'rb') as f:
    test_dataset = pickle.load(f)

#batchCosts = np.nan * torch.ones(num_test_samples)
batchCosts = np.nan * torch.ones(len(test_dataset))
counter = 0
for bat in tqdm(DataLoader(test_dataset, batch_size=1), disable=opts['no_progress_bar']):
    
    chGain = bat['chGain'] #(1, n_loc)
    dataLoad = bat['dataLoad'] #(1, n_users)
    puncFlag = bat['puncFlag'] #(1, n_RBs)
    numerology = bat['numerology'] #(1)
    availRBNum = bat['availRBNum'] #(1)
    
    _, n_users = dataLoad.size()
    _, max_n_RBs = puncFlag.size()
    n_availRBs = availRBNum[0].tolist()
    
    totalNumOfAllocation = (n_users+1) ** n_availRBs # + 1 is considered for not allocating a PRB, i.e., index zero 
    
    if n_availRBs == 0:
        batchCosts[counter] =  n_users
        print(batchCosts[counter])
        counter += 1
        continue
    
    chDispersionPerRB = 1. - (1. / ((1. + SNR_THR)**2))
    chUse = torch.tensor(CHANNEL_USE[numerology[0].tolist()])
    qFuncInv = math.sqrt(2) * erfinv(1- (2 * EPSILON))
            
            
    dataRatePerRB = (1./math.log(2)) * (chUse * (math.log(1 + SNR_THR)) - math.sqrt(chUse * chDispersionPerRB) * qFuncInv)
    
    minCost = torch.tensor([1e10]) # Initialization with a large number
    
    for ii in range(totalNumOfAllocation):
        
        dataRate=torch.zeros_like(dataLoad).squeeze(0)
        remainingDemands=dataLoad.clone().squeeze(0)
        remainingUnusedRB= availRBNum - torch.count_nonzero(puncFlag,dim=-1)
        consumedPower = torch.zeros(1, dtype=torch.float, device=chGain.device)
        puncRBsNum=torch.zeros(1, dtype=torch.int64, device=chGain.device)
        
        userPRB_alloc = torch.tensor(DecimalToAnyBaseArray(ii, n_users+1))
        userPRB_alloc_tot = torch.cat((torch.zeros(n_availRBs-len(userPRB_alloc)) ,userPRB_alloc), 0).to(torch.int)
        
        RequiredRBsForUsers = (dataLoad/dataRatePerRB).squeeze().ceil()
        
        nonFeasibleSolFlag = False
        

        for user in range(n_users):
            PRB_idxForUser = torch.nonzero(userPRB_alloc_tot == (user+1)).squeeze(1)
            
            if (RequiredRBsForUsers[user] < len(PRB_idxForUser)) | (nonFeasibleSolFlag == True):
                nonFeasibleSolFlag = True
                continue
            
            if len(PRB_idxForUser) > 0:
                
                
                puncRBsNum += torch.count_nonzero(puncFlag[0,PRB_idxForUser])
                remainingUnusedRB -= torch.count_nonzero(1-puncFlag[0,PRB_idxForUser])
                
                selected_locs = max_n_RBs * user + PRB_idxForUser
                selected_chGain = chGain[0, selected_locs]
                allocatedPower = SNR_THR/selected_chGain
                #consumedPower += allocatedPower.sum()
                consumedPower += 10.0 * torch.log10(allocatedPower).sum()
                
            
                chDispersion = 1. - (1. / ((1. + allocatedPower * selected_chGain)**2))
                chUse = torch.tensor(CHANNEL_USE[numerology[0].tolist()])
                qFuncInv = math.sqrt(2) * erfinv(1- (2 * EPSILON))
            
            
                dataRate[torch.tensor([user])] += (1./math.log(2)) * (chUse * (torch.log(1 + allocatedPower * selected_chGain)) -
                                                                       torch.sqrt(chUse * chDispersion) * qFuncInv).sum()
                remainingDemands[torch.tensor([user])] -= dataRate[torch.tensor([user])]
                remainingDemands[torch.tensor([user])] = torch.max(remainingDemands[torch.tensor([user])],torch.tensor([0]))
                
        totalUnusedAvailRBs = availRBNum - torch.count_nonzero(puncFlag,dim=-1)
        totalUsedRBs = (totalUnusedAvailRBs - remainingUnusedRB) + puncRBsNum
        
        # is not feasible solution if there is still some RBs available and some of users that their demands are not met
        if ((totalUsedRBs < availRBNum) & (remainingDemands.sum(-1) > 0)) | (nonFeasibleSolFlag == True):
            continue
        
        iter_cost = consumedPower/60.0 + torch.count_nonzero((remainingDemands>0).float(),dim=-1) + ((remainingUnusedRB > 0) & (puncRBsNum > 0)).float()
        if iter_cost < minCost:
            minCost = iter_cost
            
    batchCosts[counter] = minCost 
    print(batchCosts[counter])
    counter += 1
            
            
            
            
            
            
            
        