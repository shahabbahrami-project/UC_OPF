import torch 
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter
from scipy.special import erfinv
import math


class StateUC(NamedTuple):
    # Fixed input
    ck0: torch.Tensor 
    ck1: torch.Tensor 
    ck2: torch.Tensor 
    tu: torch.Tensor
    td: torch.Tensor 
    pMin: torch.Tensor
    pMax: torch.Tensor
    totalLoad: torch.Tensor
    totalqLoad: torch.Tensor   

    numGen:torch.Tensor
    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows
    
    # State
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    remainingGen: torch.Tensor

    i: torch.Tensor  # Keeps track of step
    
       
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
                remainingGen=self.remainingGen[key],
            )
        return super(StateUC, self).__getitem__(key)
    
    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):
        
        ck0 = input['ck0'] #(batch_size, n_loc)
        ck1 = input['ck1'] #(batch_size, n_users)
        ck2 = input['ck2'] #(batch_size, n_RBs)
        tu = input['tu'] #(batch_size)
        td = input['td'] #(batch_size)
        pMin = input['pMin'] #(batch_size)
        pMax = input['pMax'] #(batch_size)
        totalLoad = input['totalLoad']
        totalqLoad = input['totalqLoad']

        batch_size, n_gen = ck0.size()
        n_loc=2*n_gen
        
        numGen=n_gen
        
        return StateUC(
            
            ck0=ck0,
            ck1=ck1,
            ck2=ck2,
            tu=tu,
            td=td,
            pMin=pMin,
            pMax=pMax,
            totalLoad= totalLoad,
            totalqLoad= totalqLoad,
            numGen=numGen,
            
            ids=torch.arange(batch_size, dtype=torch.int64, device=ck0.device),
            prev_a=torch.zeros(batch_size, dtype=torch.long, device=ck0.device),
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, n_loc + 1,
                    dtype=torch.uint8, device=ck0.device
                )
                if visited_dtype == torch.uint8
                else print("You should complete this section") #torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            remainingGen= ngen*torch.ones(batch_size,1, dtype=torch.int64, device=ck0.device), # number of available RBs 
            
            i=torch.zeros(1, dtype=torch.int64, device=ck0.device)  # Vector with length num_step
        )
    
    def get_remaining_Gens(self):
        # returns the remaining resource blocks
        remainingGens = self.remainingGen
        return remainingGens[:, None]  # (batch_size, 1) # Add dimension for step
    
    
    def get_num_Gens(self):
        # returns the remaining resource blocks
        numGens = self.numGen
        return numGens  # (batch_size, 1) # Add dimension for step



    def update(self, selected):
        
        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        prev_a = selected
        
        
        ids_withoutDepot = self.ids[selected >= 1]
        
        _, n_gen = self.ck0.size()
        
        
        # selectedUsers = (selected[ids_withoutDepot] - 1) // n_RBs # -1 because of depot
        # selectedRBs = (selected[ids_withoutDepot] - 1) % n_RBs   # -1 because of depot
        visited_1_index = 1 + ((selected[ids_withoutDepot]-1) % n_gen) 
        visited_2_index= 1 + n_gen+((selected[ids_withoutDepot]-1) // n_gen) 
 

        remainingGen = self.remainingGen
        remainingGen[ids_withoutDepot] -= 1
        
        
        if self.visited_.dtype == torch.uint8:
            visited_ = self.visited_.scatter(-1, visited_1_index[:, None], 1)
            visited_ = self.visited_.scatter(-1, visited_2_index[:, None], 1)
        else:
            print("You should rewrite this function") #visited_ = mask_long_scatter(self.visited_, prev_a, check_unset=False)
        
        return self._replace(
            prev_a=prev_a, visited_=visited_, 
            remainingGen=remainingGen, i=self.i + 1
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
        
        
        mask_loc_off = (self.tu[:,1:] >0)
        mask_loc_on = (self.td[:,1:] >0)
        mask_loc_updown=torch.cat((mask_loc_on, mask_loc_off), -1)
        mask_loc = visited_loc.to(mask_loc_updown.dtype) | mask_loc_updown
        
        # cannot visit the depot if there is still some RBs available 
        mask_depot = (self.remainingGen>0) 
        
        return torch.cat((mask_depot[:,None], mask_loc), -1)[:, None, :]  # (batch_size, 1) # Add dimension for step
    
    
    def construct_solutions(self, actions):
        return actions
    
    
    
    