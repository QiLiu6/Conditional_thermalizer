from torch.utils.data import Dataset
import numpy as np
import torch
import math
import pickle

## Just hardcode this for Kolmogorov fields
field_std=4.44


def get_batch_indices(num_samps,batch_size,seed=42):
    """ For a given number of samples, return a list of batch indices
        num_samps:  total number of samples (i.e. length of training/valid/test set
        batch_size: batch size
        seed:       random seed
        
        returns a list of lists, each sublist containing the indices for each batch """

    rng = torch.Generator()
    rng.manual_seed(seed)

    idx=torch.randperm(num_samps,generator=rng)
    ## Break indices into lists of length batch size
    batches=[]
    for aa in range(0,num_samps,batch_size):
        batches.append(idx[aa:aa+batch_size])
    return batches


def parse_data_file(config):
    """ From a config dict, this function will:
    1. load and normalise data from the file path
    2. Split into train and valid splits (can use a fixed seed for this)
    3.Update config dictionary with metadata

    Returns tuple of train_data, valid_data, config dict
    where train and valid data are torch tensors
    """
    with open(config["file_path"], "rb") as input_file:
        data = pickle.load(input_file)
    data_config=data["data_config"]

    ## Subsample data if requested
    if "subsample" in config.keys():
        if config["subsample"] is not None:
            data=data["data"][:config["subsample"]]
        else:
            data=data["data"]

    ## Set seed for train/valid splits
    if "seed" in config.keys():
        seed=config["seed"]
    else:
        seed=42

    if config.get("train_ratio"):
        train_ratio=config["train_ratio"]
    else:
        train_ratio=0.75
        
    ## Get train/valid splits & cut data
    train_idx,valid_idx=get_split_indices(len(data),seed,train_ratio)
    train_data=data[train_idx]/field_std
    valid_data=data[valid_idx]/field_std

    ## Update config dict with data config
    for key in data_config.keys():
        config[key]=data_config[key]

    config["rollout"]=data.shape[1]
    config["field_std"]=field_std
    config["train_fields"]=len(train_data)
    config["valid_fields"]=len(valid_data)
    
    return train_data, valid_data, config

def parse_data_file_qg(config):
    """ From a config file, load the corresponding torch tensor. Split into
    train and validation tensors, and update config dict with metadata """

    ## If eddy vs jet config isn't in the dict, identify from the file path
    ## and update dict
    if ("qg" in config.keys()) == False:
        if "eddy" in config["file_path"]:
            config["qg"]="eddy"
        elif "jet" in config["file_path"]:
            config["qg"]="jet"
        else:
            print("Neither jet nor eddy identified")

    ## Use fixed normalisation for QG eddy and jet configs
    if config["qg"]=="eddy":
        upper_std=8.6294e-06 
        lower_std=1.1706e-06
    elif config["qg"]=="jet":
        upper_std=7.6382e-06
        lower_std=3.0120e-07
    with open(config["file_path"], "rb") as input_file:
        data = torch.load(input_file)

    if len(data.shape)==5:
        data[:,:,0,:,:]/=upper_std
        data[:,:,1,:,:]/=lower_std
    else:
        data[:,0,:,:]/=upper_std
        data[:,1,:,:]/=lower_std

    ## Subsample data if requested
    if "subsample" in config.keys():
        if config["subsample"] is not None:
            data=data[:config["subsample"]]

    ## Set seed for train/valid splits
    if "seed" in config.keys():
        seed=config["seed"]
    else:
        seed=42

    if config.get("train_ratio"):
        train_ratio=config["train_ratio"]
    else:
        train_ratio=0.75
        
    ## Get train/valid splits & cut data
    train_idx,valid_idx=get_split_indices(len(data),seed,train_ratio)
    train_data=data[train_idx]
    valid_data=data[valid_idx]

    config["rollout"]=data.shape[1]
    config["upper_std"]=upper_std
    config["lower_std"]=lower_std
    config["train_fields"]=len(train_data)
    config["valid_fields"]=len(valid_data)
    
    return train_data, valid_data, config

def get_split_indices(set_size,seed=42,train_ratio=0.75):
    """ Get indices for train, valid and test splits """

    valid_ratio=1-train_ratio
    
    rng = np.random.default_rng(seed)
    ## Randomly shuffle indices of entire dataset
    rand_indices=rng.permutation(np.arange(set_size))

    ## Set number of train, valid and test points
    num_train=math.floor(set_size*train_ratio)
    num_valid=math.floor(set_size*valid_ratio)
    
    ## Make sure we aren't overcounting
    assert (num_train+num_valid) <= set_size
    
    ## Pick train, test and valid indices from shuffled list
    train_idx=rand_indices[0:num_train]
    valid_idx=rand_indices[num_train+1:num_train+num_valid]
    
    ## Make sure there's no overlap between train, valid and test data
    assert len(set(train_idx) & set(valid_idx))==0, (
            "Common elements in train, valid or test set")
    return train_idx, valid_idx


class BaseDataset(Dataset):
    """ Base object to store core dataset methods """
    def __init__(self,seed=42,subsample=None,drop_spin_up=True,train_ratio=0.75,valid_ratio=0.25,test_ratio=0.0):
        super().__init__()
        self.train_ratio=train_ratio
        self.valid_ratio=valid_ratio
        self.test_ratio=test_ratio
        self.subsample=subsample
        self.seed=seed
        self.rng = np.random.default_rng(self.seed)

    def _get_split_indices(self):
        """ Set indices for train, valid and test splits """

        ## Randomly shuffle indices of entire dataset
        rand_indices=self.rng.permutation(np.arange(self.len))

        ## Set number of train, valid and test points
        num_train=math.floor(self.len*self.train_ratio)
        num_valid=math.floor(self.len*self.valid_ratio)
        num_test=math.floor(self.len*self.test_ratio)
        
        ## Make sure we aren't overcounting
        assert (num_train+num_valid+num_test) <= self.len
        
        ## Pick train, test and valid indices from shuffled list
        self.train_idx=rand_indices[0:num_train]
        self.valid_idx=rand_indices[num_train+1:num_train+num_valid]
        self.test_idx=rand_indices[len(self.valid_idx)+1:]
        
        ## Make sure there's no overlap between train, valid and test data
        assert len(set(self.train_idx) & set(self.valid_idx) & set(self.test_idx))==0, (
                "Common elements in train, valid or test set")
        return

    def _subsample(self):
        """ Take a subsample of the loaded data. Update len """
        self.x_data=self.x_data[:self.subsample]
        self.len=len(self.x_data)
        return

    def __len__(self):
        return self.len


class KSDataset(BaseDataset):
    """
    Dataset for Kuramoto-Sivashinky solutions
    """
    def __init__(self,file_path,seed=42,subsample=None,train_ratio=0.75,valid_ratio=0.25,test_ratio=0.0):
        """
        file_path:     path to data
        seed:          random seed used to create train/valid/test splits
        subsample:     None or int: if int, subsample the dataset to a total of N=subsample maps
        train_ratio:
        valid_ratio:
        test_ratio:
        """
        super().__init__(subsample=subsample,seed=seed,train_ratio=train_ratio,valid_ratio=valid_ratio,test_ratio=test_ratio)
        self.file_path=file_path
        if isinstance(self.file_path,str):
            self.x_data=torch.load(self.file_path)
        else:
            self.x_data=self.file_path
        self.len=len(self.x_data)

        if self.subsample:
            self._subsample()
        self._get_split_indices()
            
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x_data[idx]


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
        tensor containing data - assume that this is already normalised
        """
        
        super().__init__()
        self.x_data=data_tensor
        self.len=len(self.x_data)

    def __len__(self):
        return self.len
            
    def __getitem__(self, idx):
        """ Return elements at each index specified by idx."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x_data[idx]
