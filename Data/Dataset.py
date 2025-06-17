from torch.utils.data import Dataset
import numpy as np
import torch
import math
import pickle

# Just hardcode this for Kolmogorov fields
field_std=4.44
 
 
def get_batch_indices(num_samps,batch_size,seed=67):
    """ 
    For a given number of samples, return a list of batch indices
    
    Inputs:
        num_samps: int :  total number of samples (i.e. length of training/valid/test set)
        batch_size: int : batch size
        seed: int :       random seed
    Outputs:
        batches: list of lists : each sublist containing the indices for each batch 
    """

    rng = torch.Generator()
    rng.manual_seed(seed)

    #Returns a list of integers less than num_samps in shuffled order
    idx = torch.randperm(num_samps,generator=rng)

    # Break indices into lists of length batch size
    batches = []
    for j in range(0,num_samps,batch_size):
        batches.append(idx[j:j+batch_size])
    return batches


def parse_data_file(config):
    """ 
    From a config dict, this function will:
    1. load and normalise data from the file path
    2. Split into train and valid splits (can use a fixed seed for this)
    3.Update config dictionary with metadata

    Input:
        config: dictionary : a configaration dictionary that is used across the model
    Output:
        train_data: torch tensors: data used for training
        valid_data: torch tensors: data used for validation
        config dict: dictionary : updated dictionary
    """
    # the config dictionary needs to have a key "file path" and the value as where the data is saved
    loaded_data = torch.load(config["file_path"])
    data = loaded_data["data"]

    #creating a local variable in the method, config dictionary must have "data_config" as key and a subditionary as value
    data_config = loaded_data["data_config"]

    # Set seed for train/valid splits
    if "seed" in data_config.keys():
        seed = data_config["seed"]
    else:
        seed = 67

    # set training / validation ratio
    if data_config.get("train_ratio"):
        train_ratio = data_config["train_ratio"]
    else:
        train_ratio = 0.75

    # Get train/valid splits & cut data ( len(data) returns the size of the first dimension of the torch.tensor)
    train_idx,valid_idx=get_split_indices(len(data),seed,train_ratio)
    # data[int] is the same as data[int,:,:,:] for (n-1) :s, where n is the total dimension of the tensor
    train_data = data[train_idx]/field_std
    valid_data = data[valid_idx]/field_std

    # Update config dict with data config
    for key in data_config.keys():
        if key not in ("file_path", "save_path"):
            config[key] = data_config[key]

    config["rollout"]=data.shape[1]
    config["field_std"]=field_std
    #returns the number of training and validation rollouts
    config["train_fields"]=len(train_data)
    config["valid_fields"]=len(valid_data)

    return train_data, valid_data, config
    
def get_split_indices(set_size,seed=67,train_ratio=0.75):
    """
    Get indices for train and valid splits 
    
    Inputs:
        set_size: int:    the size of the first dimension of the tensor
        seed: int:        rand seed
        train_ratio: int: ratio to split up the indices by
    """

    valid_ratio=1-train_ratio

    rng = np.random.default_rng(seed)
    ## Randomly shuffle indices of entire dataset
    rand_indices=rng.permutation(np.arange(set_size))

    ## Set number of train and validation points
    num_train = math.floor(set_size*train_ratio)
    num_valid = math.floor(set_size*valid_ratio)

    ## Make sure we aren't overcounting
    assert (num_train+num_valid) <= set_size

    ## Pick train and valid indices from shuffled list
    train_idx=rand_indices[0:num_train]
    valid_idx=rand_indices[num_train+1:num_train+num_valid]

    ## Make sure there's no overlap between train and valid set
    assert len(set(train_idx) & set(valid_idx))==0, (
            "Common elements in train or valid set")
    return train_idx, valid_idx

class FluidDataset(Dataset):
    """
    Dataset for fluid flow trajectories
    Can work with either QG or Kolmogorov - it is
    agnostic to the number of input channels and input
    normalisations etc. All this processing is done in 
    the parse_data_file functions.
    """
    def __init__(self,data_tensor):
        """
        constructor: 

        Input:
            data_tensor: torch.tensor: tensor containing data - assume that this is already normalised
        """
        super().__init__()
        self.x_data = data_tensor
        self.len = len(self.x_data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """ Return elements at each index specified by idx."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x_data[idx]
