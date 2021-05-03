import torch 
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter
from scipy.special import erfinv
import math

class StateLL(NamedTuple):
    # Fixed input
    chCond: torch.Tensor # channel gain of depot + channel gain of (user,RB) pairs
    dataLoad: torch.Tensor 
    puncFlag: torch.Tensor
    numerology: torch.Tensor
    availRBNum: torch.Tensor # Number of RBs which are available for allocation 
    
    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    cur_chCond: torch.Tensor
    dataRate: torch.Tensor
    remainingDemands: torch.Tensor  # Keeps track of remaining data load
    consumedPower: torch.Tensor # Keeps track of consumed power
    remainingUnusedRB: torch.Tensor
    puncRBsNum: torch.Tensor
    i: torch.Tensor  # Keeps track of step
    
    #TX_POWER = 10  # Hardcoded
    
    SNR_THR_DB = 5  # Hardcoded  SNR threshold for URLLC users in db
    SNR_THR = 10**(SNR_THR_DB/10)
    
    CHANNEL_USE = {
        0: 24.,
        1: 48.,
        2: 96.,
        }
    
    EPSILON = 1e-4
    
    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            print("You should rewrite this function") #return mask_long2bool(self.visited_, n=self.demand.size(-1))
            
    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                prev_a=self.prev_a[key],
                visited_=self.visited_[key],
                cur_chCond=self.cur_chCond[key],
                dataRate=self.dataRate[key],
                remainingDemands=self.remainingDemands[key],
                consumedPower=self.consumedPower[key],
                remainingUnusedRB=self.remainingUnusedRB[key],
                puncRBsNum=self.puncRBsNum[key],
            )
        return super(StateLL, self).__getitem__(key)

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):
        
        chGain = input['chGain'] #(batch_size, n_loc)
        dataLoad = input['dataLoad'] #(batch_size, n_users)
        puncFlag = input['puncFlag'] #(batch_size, n_RBs)
        numerology = input['numerology'] #(batch_size)
        availRBNum = input['availRBNum'] #(batch_size)
        
        batch_size, n_loc = chGain.size()
        _, n_users = dataLoad.size()
        _, n_RBs = puncFlag.size()
        
        assert n_users*n_RBs == n_loc
        
        depotChGain = torch.zeros(batch_size, dtype=torch.float, device=chGain.device) #(batch_size)
        chCond = torch.cat((depotChGain[:, None], chGain), -1) #(batch_size, n_loc+1)
        
        return StateLL(
            
            chCond=chCond,
            dataLoad=dataLoad,
            puncFlag=puncFlag,
            numerology=numerology,
            availRBNum=availRBNum,
            ids=torch.arange(batch_size, dtype=torch.int64, device=chGain.device),
            prev_a=torch.zeros(batch_size, dtype=torch.long, device=chGain.device),
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, n_loc + 1,
                    dtype=torch.uint8, device=chGain.device
                )
                if visited_dtype == torch.uint8
                else print("You should complete this section") #torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            cur_chCond=depotChGain,
            dataRate=torch.zeros_like(dataLoad),
            remainingDemands=dataLoad.clone(), 
            consumedPower = torch.zeros(batch_size, dtype=torch.float, device=chGain.device),
            remainingUnusedRB= availRBNum - torch.count_nonzero(puncFlag,dim=-1), # number of available RBs which are not used by eMBB users
            puncRBsNum=torch.zeros(batch_size, dtype=torch.int64, device=chGain.device),
            i=torch.zeros(1, dtype=torch.int64, device=chGain.device)  # Vector with length num_step
        )
    
    def get_remaining_RBs(self):
        # returns the remaining resource blocks
        remainingRBs = torch.count_nonzero(self.puncFlag,dim=-1) + self.remainingUnusedRB - self.puncRBsNum
        return remainingRBs[:, None]  # (batch_size, 1) # Add dimension for step
    
    def get_remaining_demands(self):
        # returns the remaining demands
        batch_size, n_RBs = self.puncFlag.size()
        userChPairRemainingDemand = self.remainingDemands.repeat_interleave(n_RBs, dim=-1)
        return torch.cat((
            userChPairRemainingDemand.new_zeros(batch_size, 1),
            userChPairRemainingDemand[:, :]
            ), 1)[:, None, :]  # (batch_size, 1) # Add dimension for step
    
    
    def get_final_cost(self):
        
        assert self.all_finished()
        
        #totalUnusedAvailRBs = self.availRBNum - torch.count_nonzero(self.puncFlag,dim=-1)
        #totalUsedRBs = (totalUnusedAvailRBs - self.remainingUnusedRB) + self.puncRBsNum
        #cost1 = totalUsedRBs * self.TX_POWER
        cost1 = self.consumedPower
        cost2 = torch.count_nonzero((self.remainingDemands>0).float(),dim=-1)
        cost3 = ((self.remainingUnusedRB > 0) & (self.puncRBsNum > 0)).float()
        
        return cost1 + cost2 + cost3
    
    def update(self, selected):
        
        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        prev_a = selected
        
        
        cur_chCond = self.chCond[self.ids, selected]
        
        ids_withoutDepot = self.ids[selected >= 1]
        
        _, n_users = self.dataLoad.size()
        _, n_RBs = self.puncFlag.size()
        
        selectedUsers = (selected[ids_withoutDepot] - 1) // n_RBs # -1 because of depot
        selectedRBs = (selected[ids_withoutDepot] - 1) % n_RBs   # -1 because of depot
        
        ids_selectedPuncRBs = ids_withoutDepot[self.puncFlag[ids_withoutDepot, selectedRBs]==1]
        ids_selectedUnusedRBs = ids_withoutDepot[self.puncFlag[ids_withoutDepot, selectedRBs]==0]
        
        puncRBsNum = self.puncRBsNum
        puncRBsNum[ids_selectedPuncRBs] +=1
        
        remainingUnusedRB = self.remainingUnusedRB
        remainingUnusedRB[ids_selectedUnusedRBs] -= 1
        
        
        if self.visited_.dtype == torch.uint8:
            visited_ = self.visited_.scatter(-1, prev_a[:, None], 1)
            
            users_startLoc = n_RBs * torch.arange(n_users, dtype=torch.int64, device=self.chCond.device)
            mask_loc_RBs = torch.add(users_startLoc, selectedRBs[:,None]) + 1
            visited_[ids_withoutDepot,:] = visited_[ids_withoutDepot,:].scatter(-1, mask_loc_RBs, 1)
        else:
            print("You should rewrite this function") #visited_ = mask_long_scatter(self.visited_, prev_a, check_unset=False)
        
        allocatedPower = self.SNR_THR/cur_chCond[ids_withoutDepot]
        consumedPower = self.consumedPower
        consumedPower[ids_withoutDepot] += allocatedPower
        
        chDispersion = 1. - (1. / ((1. + allocatedPower * cur_chCond[ids_withoutDepot])**2))
        #chUse = torch.tensor([self.CHANNEL_USE[num] for num in self.numerology[ids_withoutDepot].tolist()])
        chUse = torch.tensor(list(map(self.CHANNEL_USE.get, self.numerology[ids_withoutDepot].tolist()))).to(self.chCond.device)    
        qFuncInv = math.sqrt(2) * erfinv(1- (2 * self.EPSILON))
        
        dataRate = self.dataRate
        dataRate[ids_withoutDepot,selectedUsers] += (1./math.log(2)) * (chUse * (torch.log(1 + allocatedPower * cur_chCond[ids_withoutDepot])) -
                                                                       torch.sqrt(chUse * chDispersion) * qFuncInv)
        
        remainingDemands = self.remainingDemands
        remainingDemands[ids_withoutDepot,selectedUsers] -= dataRate[ids_withoutDepot,selectedUsers]
        remainingDemands[ids_withoutDepot,selectedUsers] = torch.max(remainingDemands[ids_withoutDepot,selectedUsers],torch.tensor([0]).to(self.chCond.device))
        
        return self._replace(
            prev_a=prev_a, visited_=visited_, cur_chCond=cur_chCond,
            dataRate=dataRate, remainingDemands=remainingDemands, consumedPower=consumedPower,
            remainingUnusedRB=remainingUnusedRB, puncRBsNum=puncRBsNum, i=self.i + 1
            )
    
    def all_finished(self):       
        
        # All must be returned to depot (and at least 1 step since at start also prev_a == 0)
        return self.i.item() > 0 and (self.prev_a == 0).all()
    
    def get_current_node(self):
        """
        Returns the current node where 0 is depot, 1...n are nodes
        :return: (batch_size, num_steps) tensor with current nodes
        """
        return self.prev_a[:, None]  # (batch_size, 1) # Add dimension for step
        
    def get_mask(self):
        
            
        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, 1:]
        else:
            print("You should rewrite this function") #visited_loc = mask_long2bool(self.visited_, n=self.demand.size(-1))
        
        _, n_RBs = self.puncFlag.size()
        
        mask_loc_demands = (self.remainingDemands.repeat_interleave(n_RBs, dim=-1) <= 0)
        
        mask_loc_chGain = (self.chCond[:,1:] < self.EPSILON/1e10)
        
        mask_loc = visited_loc.to(mask_loc_demands.dtype) | mask_loc_demands | mask_loc_chGain
        
        
        
        totalUnusedAvailRBs = self.availRBNum - torch.count_nonzero(self.puncFlag,dim=-1)
        totalUsedRBs = (totalUnusedAvailRBs - self.remainingUnusedRB) + self.puncRBsNum
        
        # cannot visit the depot if there is still some RBs available and some of users that their demands are not met
        mask_depot = (totalUsedRBs < self.availRBNum) & (self.remainingDemands.sum(-1) > 0)
        
        return torch.cat((mask_depot[:,None], mask_loc), -1)[:, None, :]  # (batch_size, 1) # Add dimension for step
    
    
    def construct_solutions(self, actions):
        return actions
        
        
        
        
        
        
        
        
        
        