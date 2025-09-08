import torch_qg.model as torch_model
import torch_qg.parameterizations as torch_param
from tqdm import tqdm
import json
import xarray as xr
import torch

def run_test_sim(steps,hr_model=None,lr_model=None,sampling_freq=10,jet=False):
    """ Run a qg simulation trajectory. We simulate in high res, and downsample to low res. We return
        downsampled fields, sampled at `sampling_freq` intervals in numerical timestep (because we generally
        emulate in different timesteps to numerical.

        Unfortunately torch_qg converts the tensors into xarrays, and we then convert this back into torch.
        If it becomes annoying can modify torch_qg, but will stick with this until then. """

    ## Prepare dictionary in case jet config is chosen
    ## If false, we just leave this dict empty
    flow_config={}
    if jet==True:
        flow_config={'rek': 7e-08, 'delta': 0.1, 'beta': 1e-11}
    if hr_model==None:
        hr_model=torch_model.PseudoSpectralModel(nx=256,dt=3600,dealias=True,parameterization=torch_param.Smagorinsky(),**flow_config)
    if lr_model==None:
        lr_model=torch_model.PseudoSpectralModel(nx=64,dt=3600,dealias=True,**flow_config)
    ds=[]

    ## Run spinup
    for aa in range(55990):
        hr_model._step_ab3()

    ## Run physical trajectory and save snapshots at whatever selected frequency    
    for aa in tqdm(range(steps)):
        hr_model._step_ab3()
        if (aa % sampling_freq == 0):
            ds.append(hr_model.forcing_dataset(lr_model))
            
    ds=xr.concat(ds,dim="time")
    return torch.tensor(ds.q.values,dtype=torch.float32)

