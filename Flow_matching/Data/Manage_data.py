'''
Python3
Author: Qi Liu
This is a script for managing data for flow matching training so that it can be loaded easierly into the trainer.
To run by command: python3 /path/to/Manage_data.py 1.(path to emulator data) 2.(path to numerical data) 3.(path to save training data)
'''

import torch
import os
import sys

emu_path = sys.argv[1]
num_path = sys.argv[2]
out_path = sys.argv[3]

emu_data = torch.load(emu_path)
B,T,L,_ = emu_data.shape 

num_dict = torch.load(num_path)
num_data = num_dict['data'][0:B,0:T]
data_config = num_dict['data_config']

# Train with Gaussian first
#num_data = torch.rand_like(emu_data)
    
training_data = torch.cat((emu_data,num_data),dim=1)

output_dict = {
    'data': training_data,
    'data_config': data_config
}

# Save the dictionary
torch.save(output_dict, out_path)

print(f"Saved combined data to {out_path}")