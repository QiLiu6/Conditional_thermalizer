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
import Data.Data_generation.Kolmogorov.util as util
from tqdm import tqdm


####
### Functions to run emulator and thermalize trajectories
###
def run_conditional_emu(ics, emu, therm=None, n_steps=1000, lag = torch.tensor([1]), denoising_steps=5, freq = 25, silent=True, sigma=None, Regression = True):
    """ Run an emuluator on some ICs
    inputs:   
        ics: Bx1xLxL tensor:    initial conditions for emulator
        emu: nn.Module       :    torch emulator model
        therm:  nn.Module    :    diffusion model object
        n_steps: int         :    how many emulator steps to run
        lag: int             :    how many snapshots in the past to condition on, needs to be a uniform tensor with the same length as the batch size of ics
        denoising_steps: int :    how many denoising steps to take
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
    if Regression == True:
        with torch.no_grad(): 
            for aa in tqdm(range(1,n_steps),disable=silent):
                state_vector[:,aa]=emu(state_vector[:,aa-1].unsqueeze(1)).squeeze()+state_vector[:,aa-1]
                if sigma:
                    state_vector[:,aa]+=sigma*torch.randn_like(state_vector[:,aa],device=state_vector[:,aa].device)
                if therm and aa % freq == 0 and aa - freq * lag[0].item() > 0:
                    x_t_noised = state_vector[:,aa].unsqueeze(1)
                    x_t_minus = state_vector[:,aa - freq * lag[0].item()].unsqueeze(1)
                    noised_plus_conditional = torch.cat((x_t_noised, x_t_minus), dim=1)
                    pred_noise, pred_noise_level = therm.model(noised_plus_conditional,lag[0].item(),True)
                    pred_noise = pred_noise.to("cuda")
                    pred_noise_level = pred_noise_level.to("cuda")
                    state_vector[:,aa]=therm.denoising(noised_plus_conditional, lag, pred_noise_level).squeeze()
        enstrophies=(abs(state_vector**2).sum(axis=(2,3)))

    else:
        with torch.no_grad(): 
            for aa in tqdm(range(1,n_steps),disable=silent):
                state_vector[:,aa]=emu(state_vector[:,aa-1].unsqueeze(1)).squeeze()+state_vector[:,aa-1]
                if sigma:
                    state_vector[:,aa]+=sigma*torch.randn_like(state_vector[:,aa],device=state_vector[:,aa].device)
                if therm and aa % freq == 0 and aa - freq * lag[0].item() > 0:
                    x_t_noised = state_vector[:,aa].unsqueeze(1)
                    x_t_minus = state_vector[:,aa - freq * lag[0].item()].unsqueeze(1)
                    noised_plus_conditional = torch.cat((x_t_noised, x_t_minus), dim=1)
                    state_vector[:,aa]=therm.denoising(noised_plus_conditional, lag, denoising_steps).squeeze()
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
            state_vector[:,aa]=emu(state_vector[:,aa-1].unsqueeze(1)).squeeze()+state_vector[:,aa-1]
            if sigma:
                state_vector[:,aa]+=sigma*torch.randn_like(state_vector[:,aa],device=state_vector[:,aa].device)
    return state_vector