import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.animation as animation
from IPython.display import HTML
import os
import sys
sys.path.append('/home/ql2221/Projects/thermalizer/thermalizer/kolmogorov')
import util
from tqdm import tqdm


####
### Functions to run emulator and thermalize trajectories using flow matching
###
def run_flow_matching_emu(ics, emu, flow=None, n_steps=1000, silent=True, sigma=None):
    """ Run an emuluator on some ICs
    inputs:   
        ics: Bx1xLxL tensor:    initial conditions for emulator
        emu: nn.Module       :    torch emulator model
        therm:  nn.Module    :    flow matching nn
        n_steps: int         :    how many emulator steps to run
        lag: int             :    how many snapshots is the drift seperated from the initial conditions
        silent: boolean      :    silence tqdm progress bar (for slurm scripts)
        sigma:  int          :    noise std level if we have a stochastic rollout
        
    Outputs:
        state_vector: B x n_steps x L x L tensor: rollout of the ic using our models
        enstrophies: B x n_steps x 1            : rollout flows' enstrophies
    """
    ## Set up state and diagnostic tensors
    state_vector=torch.zeros((len(ics),n_steps,64,64),device="cuda")
    ## Set ICs
    state_vector[:,0:1]=ics
    state_vector=state_vector.to("cuda")

    t = torch.ones(ics.shape[0]).to(ics.device)
    s = torch.zeros(ics.shape[0]).to(ics.device)
    with torch.no_grad(): 
        for aa in tqdm(range(1,n_steps),disable=silent):
            state_vector[:,aa]=emu(state_vector[:,aa-1].unsqueeze(1)).squeeze()+state_vector[:,aa-1]
            if sigma:
                state_vector[:,aa]+=sigma*torch.randn_like(state_vector[:,aa],device=state_vector[:,aa].device)
            if flow:  
                x_t_noised = state_vector[:,aa].unsqueeze(1)
                x_t_prime = flow(x_t_noised, t, s)
                state_vector[:,aa:aa+1] += x_t_prime
    enstrophies=(abs(state_vector**2).sum(axis=(2,3)))
    return state_vector, enstrophies

def run_emu(ics,emu,n_steps=1270,silent=False,sigma=None):
    """ Run an emuluator on some ICs
        ics:     initial conditions for emulator
        emu:     torch emulator model
        n_steps: how many emulator steps to run
        silent:  silence tqdm progress bar (for slurm scripts)
        sigma:   noise std level if we have a stochastic rollout """
    ## Set up state tensors
    state_vector=torch.zeros((len(ics),n_steps,64,64),device="cuda")
    
    ## Set ICs
    state_vector[:,0:1]=ics
    state_vector=state_vector.to("cuda")

    with torch.no_grad(): 
        for aa in tqdm(range(1,n_steps),disable=silent):
            state_vector[:,aa:aa+1]=emu(state_vector[:,aa-1:aa])+state_vector[:,aa-1:aa]
            if sigma:
                state_vector[:,aa]+=sigma*torch.randn_like(state_vector[:,aa],device=state_vector[:,aa].device)
    return state_vector