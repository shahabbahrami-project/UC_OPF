import torch 
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter
from scipy.special import erfinv
import math

class StateML(NamedTuple):
    # Fixed input
    chCond: torch.Tensor # channel gain of depot + channel gain of (user,RB) pairs
    minRateReq: torch.Tensor 
    sharedFlag: torch.Tensor 
    numerology: torch.Tensor
    availRBNum: torch.Tensor # Number of RBs which are available for allocation 
    availPower: torch.Tensor
    
    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows
    
    # State
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    remainingRB: torch.Tensor
    i: torch.Tensor  # Keeps track of step
    
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
                remainingRB=self.remainingRB[key],
            )
        return super(StateML, self).__getitem__(key)
    
    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):
        
        
        chGain = input['chGain'] #(batch_size, n_loc)
        minRateReq = input['minRateReq'] #(batch_size, n_users)
        sharedFlag = input['sharedFlag'] #(batch_size, n_RBs)
        numerology = input['numerology'] #(batch_size)
        availRBNum = input['availRBNum'] #(batch_size)
        availPower = input['availPower'] #(batch_size)
        
        batch_size, n_loc = chGain.size()
        _, n_users = minRateReq.size()
        _, n_RBs = sharedFlag.size()
        
        assert n_users*n_RBs == n_loc
        
        depotChGain = torch.zeros(batch_size, dtype=torch.float, device=chGain.device) #(batch_size)
        chCond = torch.cat((depotChGain[:, None], chGain), -1) #(batch_size, n_loc+1)
        
        return StateML(
            
            chCond=chCond,
            minRateReq=minRateReq,
            sharedFlag=sharedFlag,
            numerology=numerology,
            availRBNum=availRBNum,
            availPower=availPower,
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
            remainingRB= availRBNum.clone(), # number of available RBs 
            i=torch.zeros(1, dtype=torch.int64, device=chGain.device)  # Vector with length num_step
        )
    
    def get_remaining_RBs(self):
        # returns the remaining resource blocks
        remainingRBs = self.remainingRB
        return remainingRBs[:, None]  # (batch_size, 1) # Add dimension for step
    
    
    def update(self, selected):
        
        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        prev_a = selected
        
        
        ids_withoutDepot = self.ids[selected >= 1]
        
        _, n_users = self.minRateReq.size()
        _, n_RBs = self.sharedFlag.size()
        
        selectedUsers = (selected[ids_withoutDepot] - 1) // n_RBs # -1 because of depot
        selectedRBs = (selected[ids_withoutDepot] - 1) % n_RBs   # -1 because of depot
        
        
        remainingRB = self.remainingRB
        remainingRB[ids_withoutDepot] -= 1
        
        
        if self.visited_.dtype == torch.uint8:
            visited_ = self.visited_.scatter(-1, prev_a[:, None], 1)
            
            users_startLoc = n_RBs * torch.arange(n_users, dtype=torch.int64, device=self.chCond.device)
            mask_loc_RBs = torch.add(users_startLoc, selectedRBs[:,None]) + 1
            visited_[ids_withoutDepot,:] = visited_[ids_withoutDepot,:].scatter(-1, mask_loc_RBs, 1)
        else:
            print("You should rewrite this function") #visited_ = mask_long_scatter(self.visited_, prev_a, check_unset=False)
        
        return self._replace(
            prev_a=prev_a, visited_=visited_, 
            remainingRB=remainingRB, i=self.i + 1
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
        
        
        mask_loc_chGain = (self.chCond[:,1:] < self.EPSILON/1e10)
        
        mask_loc = visited_loc.to(mask_loc_chGain.dtype) | mask_loc_chGain
        
        totalAvailRBs = self.availRBNum 
        totalUsedRBs = (totalAvailRBs - self.remainingRB)
        
        # cannot visit the depot if there is still some RBs available 
        mask_depot = (totalUsedRBs < self.availRBNum) 
        
        return torch.cat((mask_depot[:,None], mask_loc), -1)[:, None, :]  # (batch_size, 1) # Add dimension for step
    
    
    def construct_solutions(self, actions):
        return actions
    
    
    
    