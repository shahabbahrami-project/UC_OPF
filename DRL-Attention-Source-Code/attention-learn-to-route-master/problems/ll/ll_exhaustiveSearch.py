import pickle
import torch
# Bring your packages onto the path
import sys, os
sys.path.append(os.path.abspath(os.path.join('../..')))


from utils.data_utils import load_dataset



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



file_path = os.path.abspath(os.path.join('../../outputs/ll_96/run_20210219T162115/ll96_validation_seed1234.pkl'))

data_set = []

if file_path is not None:
    assert os.path.splitext(file_path)[1] == '.pkl'
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
'''       
        data_set = [
            {
                'chGain': torch.FloatTensor(chGain),      #(batch_size, n_loc)
                'dataLoad': torch.FloatTensor(dataLoad),  #(batch_size, n_users)
                'puncFlag': torch.FloatTensor(puncFlag),  #(batch_size, n_RBs)
                'numerology': torch.tensor(numerology),   #(batch_size)
                'availRBNum': torch.tensor(availRBNum)    #(batch_size)
                }
            for chGain, dataLoad, puncFlag, numerology, availRBNum in (data[offset:offset+num_samples])
            ]

for ii in range(81):
    print(DecimalToAnyBaseArray(ii, 3))
'''