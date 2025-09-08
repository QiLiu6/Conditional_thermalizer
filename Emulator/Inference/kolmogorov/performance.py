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
### Functions to run emulator and thermalize trajectories
###
def run_emu(ics,emu,therm=None,n_steps=1000,silent=False,sigma=None):
    """ Run an emuluator on some ICs
        ics:     initial conditions for emulator
        emu:     torch emulator model
        therm:   diffusion model object - pass this if we want to
                 also classify the noise level during rollout
        n_steps: how many emulator steps to run
        silent:  silence tqdm progress bar (for slurm scripts)
        sigma:   noise std level if we have a stochastic rollout """
    ## Set up state and diagnostic tensors
    state_vector=torch.zeros((len(ics),n_steps,64,64),device="cuda")
    ## Set ICs
    state_vector[:,0:1]=ics
    state_vector=state_vector.to("cuda")
    noise_classes=None
    if therm:
        noise_classes=torch.zeros(len(state_vector),len(state_vector[1]))

    with torch.no_grad(): 
        for aa in tqdm(range(1,n_steps),disable=silent):
            state_vector[:,aa]=emu(state_vector[:,aa-1].unsqueeze(1)).squeeze()+state_vector[:,aa-1]
            if sigma:
                state_vector[:,aa]+=sigma*torch.randn_like(state_vector[:,aa],device=state_vector[:,aa].device)
            if therm:
                preds=therm.model.noise_class(state_vector[:,aa].unsqueeze(1))
                noise_classes[:,aa]=preds.cpu()
    enstrophies=(abs(state_vector**2).sum(axis=(2,3)))
    return state_vector, enstrophies, noise_classes


def therm_algo(ics,emu,therm,n_steps=-1,start=10,stop=4,forward=True,silent=False,noise_limit=100,sigma=None):
    """ Thermalization algorithm 2 - correct algo where we only thermalize elements over the
        initialisation threshold:
        ics:         initial conditions for emulator
        emu:         torch emulator model
        therm:       diffusion model object
        n_steps:     how many emulator steps to run
        start:       noise level to start thermalizing
        stop:        noise level to stop thermalizing
        forward:     Add forward diffusion noise when thermalizing
        silent:      silence tqdm progress bar (for slurm scripts)
        noise_limit: if predicted noise level exceeds this threshold, cut the run
        sigma:       noise std level if we have a stochastic rollout

        returns: state_vector, enstrophies, noise_classes, therming_counts"""

    ## Set up state and diagnostic tensors
    state_vector=torch.zeros((len(ics),n_steps,64,64),device="cuda")
    ## Set ICs
    state_vector[:,0]=ics
    noise_classes=torch.zeros(len(state_vector),len(state_vector[1]))
    therming_counts=torch.zeros_like(noise_classes)
    
    ## Run
    with torch.no_grad(): 
        for aa in tqdm(range(1,len(state_vector[1])),disable=silent):
            ## Emulator step
            state_vector[:,aa]=emu(state_vector[:,aa-1].unsqueeze(1)).squeeze()+state_vector[:,aa-1]
            if sigma:
                state_vector[:,aa]+=sigma*torch.randn_like(state_vector[:,aa],device=state_vector[:,aa].device)
            ## Noise level check
            preds=therm.model.noise_class(state_vector[:,aa].unsqueeze(1))
            noise_classes[:,aa]=preds.cpu() ## Store noise levels
            ## Indices of thermalized fields
            therm_select=preds>(start)
            therming=state_vector[therm_select,aa].unsqueeze(1)
            ## If we have any fields over threshold, run therm
            if len(therming)>0:
                thermed,counts=therm.denoise_heterogen(therming,preds[therm_select],stop=stop,forward_diff=forward)
                for bb,idx in enumerate(torch.argwhere(therm_select).flatten()):
                    state_vector[idx,aa]=thermed[bb].squeeze()
                    therming_counts[idx,aa]=counts[bb]
            if noise_limit:
                if preds.max()>noise_limit:
                    print("breaking due to noise limit at traj %d" % preds.argmax().item())
                    ## Truncate tensors to cutoff length
                    state_vector=state_vector[:,:aa+1]
                    noise_classes=noise_classes[:,:aa+1]
                    therming_counts=therming_counts[:,:aa+1]
                    break
    state_vector=state_vector.to("cpu")

    enstrophies=(abs(state_vector**2).sum(axis=(2,3)))
    return state_vector, enstrophies, noise_classes, therming_counts


def therm_algo_free(ics,emu,therm,n_steps=100,start=10,stop=4,forward=True,silent=False,noise_limit=100,sigma=None):
    """ Thermalization algorithm  - correct algo where we only thermalize elements over the
        initialisation threshold. Here we don't store the whole trajectory, so we can run for longer
        than memory requirement normally allows.
        
        Input parameters:
        ics:         initial conditions for emulator
        emu:         torch emulator model
        therm:       diffusion model object
        n_steps:     how many emulator steps to run
        start:       noise level to start thermalizing
        stop:        noise level to stop thermalizing
        forward:     Add forward diffusion noise when thermalizing
        silent:      silence tqdm progress bar (for slurm scripts)
        noise_limit: if predicted noise level exceeds this threshold, cut the run
        sigma:       noise std level if we have a stochastic rollout

        returns: state_vector,final_step_index"""

    ## Set ICs
    state_vector=ics
    
    ## Run
    with torch.no_grad(): 
        for aa in tqdm(range(1,n_steps),disable=silent):
            ## Emulator step
            state_vector=emu(state_vector.unsqueeze(1)).squeeze()+state_vector
            if sigma:
                state_vector[:,aa]+=sigma*torch.randn_like(state_vector[:,aa],device=state_vector[:,aa].device)
            ## Noise level check
            preds=therm.model.noise_class(state_vector.unsqueeze(1))
            ## Indices of thermalized fields
            therm_select=preds>(start)
            therming=state_vector[therm_select].unsqueeze(1)
            ## If we have any fields over threshold, run therm
            if len(therming)>0:
                thermed,counts=therm.denoise_heterogen(therming,preds[therm_select],stop=stop,forward_diff=forward)
                for bb,idx in enumerate(torch.argwhere(therm_select).flatten()):
                    state_vector[idx]=thermed[bb].squeeze()
            if noise_limit:
                if preds.max()>noise_limit:
                    print("breaking due to noise limit at point %d" % aa)
                    break
    return state_vector,aa


def therm_algo_batch(ics,emu,therm,n_steps=-1,start=10,stop=4,forward=True,silent=False):
    """ Thermalization algorithm batch: why do we have this algorithm? When I originally implemented thermalized
        rollouts, I included a bug which meant that all trajectories were thermalized if a single trajectory exceeded
        the initiation criteria. After fixing the bug I noticed that the stability/reliability actually decreased. So
        perhaps adding some stochastically-initiated thermalization actually helps things. Ultimately I think this all
        comes down to the inaccuracy of the noise classifier.. But either way, I keep this algorithm here just in case
        we want to come back and test this again.

    Arguments:
        ics:     initial conditions for emulator
        emu:     torch emulator model
        therm:   diffusion model object
        n_steps: how many emulator steps to run
        start:   noise level to start thermalizing
        stop:    noise level to stop thermalizing
        forward: Add forward diffusion noise when thermalizing
        silent:  silence tqdm progress bar (for slurm scripts) 
        
        returns: state_vector, enstrophies, noise_classes, therming_counts"""
    ## Set up state and diagnostic tensors
    state_vector=torch.zeros((len(ics),n_steps,64,64),device="cuda")
    ## Set ICs
    state_vector[:,0]=ics
    state_vector=state_vector.to("cuda")
    noise_classes=torch.zeros(len(state_vector),len(state_vector[1]))
    therming_counts=torch.zeros_like(noise_classes)

    with torch.no_grad(): 
        for aa in tqdm(range(1,len(state_vector[1])),disable=silent):
            state_vector[:,aa]=emu(state_vector[:,aa-1].unsqueeze(1)).squeeze()+state_vector[:,aa-1]
            preds=therm.model.noise_class(state_vector[:,aa].unsqueeze(1))
            noise_classes[:,aa]=preds.cpu()
            if max(preds)>start:
                thermed,therming_counts[:,aa]=therm.denoise_heterogen(state_vector[:,aa].unsqueeze(1),preds,stop=stop,forward_diff=forward)
                state_vector[:,aa]=thermed.squeeze()
    enstrophies=(abs(state_vector**2).sum(axis=(2,3)))
    return state_vector, enstrophies, noise_classes, therming_counts



class EmulatorRollout():
    """ Run a batch of emulator rollouts along test trajectories """
    def __init__(self,test_suite,model_emu,residual=True,sigma=None,silence=True):
        """ test_suite: torch tensor of test data with shape [batch, snapshot, Nx, Ny]. For Kolmogorov flows
                        these are not normalised, so we need to normalise using the model's field_std
            model_emu:  a torch.nn.Module emulator
            residual:   bool to determine whether our emulator predicts the state or residuals between two
                        timesteps
            sigma:      Toggle whether or not we are running a stochastic trajectory. If sigma is None, the
                        trajectory is deterministc. If sigma is a scalar, this is the variance of the noise
                        we add at each step.
            silence:    toggle for tqdm progress bar
        """
        self.test_suite=test_suite
        self.test_suite/=model_emu.config["field_std"]
        self.model_emu=model_emu
        self.residual=residual
        self.sigma=sigma
        self.silence=silence

        ## Set up field tensors
        self.emu=torch.zeros(self.test_suite.shape,dtype=torch.float32)
        ## Set t=0 to be the same
        self.emu[:,0,:,:]=self.test_suite[:,0,:,:]

        ## Ensure models are in eval
        self.model_emu.eval()
        
        if torch.cuda.is_available():
            #self.device=torch.device('cuda')
            self.device="cuda"
            ## Put models on GPU
            self.model_emu.to(self.device)
            ## Put tensors on GPU
            #self.test_suite=self.test_suite.to(self.device)
            #self.emu=self.emu.to(self.device)
        else:
            self.device="cpu"

        self.emu = self.emu.to(self.device)

        self._init_metrics()

    def _init_metrics(self):
        self.mseloss=torch.nn.MSELoss(reduction="none")
        ## Set up metric tensors
        self.mse_auto=torch.zeros(self.test_suite.shape[0],self.test_suite.shape[1])
        self.mse_emu=torch.zeros(self.test_suite.shape[0],self.test_suite.shape[1])

        self.autocorr=[]
        self.corr_emu=[]

        self.grid=util.fourierGrid(64)
        self.ke_true=torch.zeros(self.test_suite.shape[0],self.test_suite.shape[1],len(self.grid.k1d_plot))
        self.ke_emu=torch.zeros(self.ke_true.shape)

        return

    @torch.no_grad()
    def evolve(self):
        for aa in tqdm(range(1,len(self.test_suite[0])),disable=self.silence):
            ## Step fields forward
            emu_unsq=self.emu[:,aa-1,:,:].unsqueeze(1).to(self.device)
            preds=self.model_emu(emu_unsq)
            #means=torch.mean(preds,axis=(-1,-2)) ## If we wanna do zero-mean, which we don't
            if self.residual:
                self.emu[:,aa,:,:]=(preds+emu_unsq).squeeze()
                #self.emu[:,aa,:,:]=(preds-means.unsqueeze(1).unsqueeze(1)+emu_unsq).squeeze().cpu()
            else:
                self.emu[:,aa,:,:]=(preds).squeeze()
            if self.sigma: ## Add noise if we are running a stochastic trajectory
                self.emu[:,aa,:,:]+=torch.randn_like(self.emu[:,aa,:,:])*self.sigma

            ## MSE metrics
            loss=self.mseloss(self.test_suite[:,0],self.test_suite[:,aa])
            self.mse_auto[:,aa]=torch.mean(loss,dim=(1,2))
            loss=self.mseloss(self.test_suite[:,aa],self.emu[:,aa])
            self.mse_emu[:,aa]=torch.mean(loss,dim=(1,2))
            
            if self.mse_emu[:,aa].min()>10000:
                break
        
        self.emu=self.emu[:,:aa,:,:]
        self.mse_emu=self.mse_emu[:,:aa]

    def _KE_spectra(self):
        ## Move to cpu for KE spectra calculation
        self.test_suite=self.test_suite.to("cpu")
        self.emu=self.emu.to("cpu")
        self.therm=self.therm.to("cpu")
        for aa in tqdm(range(1,len(self.test_suite[1]))):
            for bb in range(0,len(self.test_suite[0])):
                _,ke=util.get_ke(self.test_suite[aa,bb],self.grid)
                self.ke_true[aa,bb]=torch.tensor(ke)
                _,ke=util.get_ke(self.emu[aa,bb],self.grid)
                self.ke_emu[aa,bb]=torch.tensor(ke)
                _,ke=util.get_ke(self.therm[aa,bb],self.grid)
                self.ke_therm[aa,bb]=torch.tensor(ke)
  
        ## Move to back to gpu
        self.test_suite=self.test_suite.to(self.device)
        self.emu=self.emu.to(self.device)
        self.therm=self.therm.to(self.device)


class ThermalizeKolmogorovDDPM():
    def __init__(self,test_suite,model_emu,model_therm,thermalize_delay=100,thermalize_interval=5,thermalize_timesteps=2):
        self.test_suite=test_suite/model_emu.config["field_std"]
        self.model_emu=model_emu
        self.model_therm=model_therm
        self.thermalize_delay=thermalize_delay
        self.thermalize_interval=thermalize_interval
        self.thermalize_timesteps=thermalize_timesteps

        ## Set up field tensors
        self.emu=torch.zeros(self.test_suite.shape)
        self.therm=torch.zeros(self.test_suite.shape)

        ## Ensure models are in eval
        self.model_emu.eval()
        self.model_therm.eval()
        
        if torch.cuda.is_available():
            self.device=torch.device('cuda')
            ## Put models on GPU
            self.model_emu=self.model_emu.to(self.device)
            self.model_therm=self.model_therm.to(self.device)
            ## Put tensors on GPU
            self.test_suite=self.test_suite.to(self.device)
            self.emu=self.emu.to(self.device)
            self.therm=self.therm.to(self.device)

        ## Set t=0 to be the same
        self.emu[:,0,:,:]=self.test_suite[:,0,:,:]
        self.therm[:,0,:,:]=self.test_suite[:,0,:,:]

        self._init_metrics()

    def _init_metrics(self):
        self.mseloss=torch.nn.MSELoss(reduction="none")
        ## Set up metric tensors
        self.mse_auto=torch.zeros(self.test_suite.shape[0],self.test_suite.shape[1])
        self.mse_emu=torch.zeros(self.test_suite.shape[0],self.test_suite.shape[1])
        self.mse_therm=torch.zeros(self.test_suite.shape[0],self.test_suite.shape[1])

        self.autocorr=[]
        self.corr_emu=[]
        self.corr_therm=[]

        self.grid=util.fourierGrid(64)
        self.ke_true=torch.zeros(self.test_suite.shape[0],self.test_suite.shape[1],len(self.grid.k1d_plot))
        self.ke_emu=torch.zeros(self.ke_true.shape)
        self.ke_therm=torch.zeros(self.ke_true.shape)

        return

    @torch.no_grad()
    def _evolve(self):
        for aa in tqdm(range(1,len(self.test_suite[1]))):
            ## Step fields forward
            emu_unsq=self.emu[:,aa-1,:,:].unsqueeze(1)
            preds=self.model_emu(emu_unsq)
            means=torch.mean(preds,axis=(-1,-2))
            self.emu[:,aa,:,:]=(preds-means.unsqueeze(1).unsqueeze(1)+emu_unsq).squeeze()

            therm_unsq=self.therm[:,aa-1,:,:].unsqueeze(1)
            preds=self.model_emu(therm_unsq)
            means=torch.mean(preds,axis=(-1,-2))
            self.therm[:,aa,:,:]=(preds-means.unsqueeze(1).unsqueeze(1)+therm_unsq).squeeze()

            should_thermalize = (aa % self.thermalize_interval == 0) and (aa>self.thermalize_delay)
            if should_thermalize:
                thermed=self.model_therm.denoising(self.therm[:,aa,:,:].unsqueeze(1),self.thermalize_timesteps)
                self.therm[:,aa,:,:]=thermed.squeeze()

            ## MSE metrics
            loss=self.mseloss(self.test_suite[:,0],self.test_suite[:,aa])
            self.mse_auto[:,aa]=torch.mean(loss,dim=(1,2))
            loss=self.mseloss(self.test_suite[:,aa],self.emu[:,aa])
            self.mse_emu[:,aa]=torch.mean(loss,dim=(1,2))
            loss=self.mseloss(self.test_suite[:,aa],self.therm[:,aa])
            self.mse_therm[:,aa]=torch.mean(loss,dim=(1,2))

    def _KE_spectra(self):
        ## Move to cpu for KE spectra calculation
        self.test_suite=self.test_suite.to("cpu")
        self.emu=self.emu.to("cpu")
        self.therm=self.therm.to("cpu")
        for aa in tqdm(range(1,len(self.test_suite[1]))):
            for bb in range(0,len(self.test_suite[0])):
                _,ke=util.get_ke(self.test_suite[aa,bb],self.grid)
                self.ke_true[aa,bb]=torch.tensor(ke)
                _,ke=util.get_ke(self.emu[aa,bb],self.grid)
                self.ke_emu[aa,bb]=torch.tensor(ke)
                _,ke=util.get_ke(self.therm[aa,bb],self.grid)
                self.ke_therm[aa,bb]=torch.tensor(ke)
  
        ## Move to back to gpu
        self.test_suite=self.test_suite.to(self.device)
        self.emu=self.emu.to(self.device)
        self.therm=self.therm.to(self.device)


class KolmogorovAnimation():
    def __init__(self,ds,model,fps=10,nSteps=1000,normalise=True,cache_residuals=False,savestring=None):
        """ Run an emulator alongside a simulated trajectory, for a RESIDUAL emulator.
            Animate the evolution of the simulated
            vs emulated vorticity fields. Have an option to cache the residuals, such that we can estimate the
            covariance of the noise in fields 
            ds contains the true simulation data. Assume that it is a NORMALISED torch tensor.
            """
        self.ds=ds
        self.model=model
        self.fps=fps
        self.nSteps=nSteps
        self.nFrames=int(self.nSteps)
        self.pred=self.ds[0]
        self.normalise=normalise
        self.cache_residuals=cache_residuals
        
        self.mse=[]
        self.correlation=[]
        self.autocorrelation=[]
        self.criterion=nn.MSELoss()
        self.times=np.arange(0,self.nFrames+0.001,1)

        self.residuals=None
        self.savestring=savestring
        self.grid=grid=util.fourierGrid(64)
        self.k1d_plot,self.ketrue=util.get_ke(self.ds[0],self.grid)
        self.k1d_plot,self.kepred=util.get_ke(self.pred,self.grid)
        
        if self.cache_residuals:
            self.residuals=np.empty(ds.shape)
        
    def _push_forward(self):
        """ Update predicted q by one emulator pass """
        
        ## Convert q to standardised q
        x=torch.tensor(self.pred).float().unsqueeze(0).unsqueeze(0)
        x=self.model(x)
        
        with torch.no_grad():
            if self.normalise==True:
                x=x-torch.mean(x)
                    
        self.pred=(self.pred+x.squeeze()).cpu()
        
        self.correlation.append(pearsonr(self.pred.flatten(),self.ds[self.i].numpy().flatten())[0])
        self.autocorrelation.append(pearsonr(self.ds[0].numpy().flatten(),self.ds[self.i].numpy().flatten())[0])
        self.mse.append(self.criterion(torch.tensor(self.pred),torch.tensor(self.ds[self.i].numpy())))

        if self.cache_residuals:
            self.residuals[self.i]=self.s[self.i]-self.q_i_pred

        _,self.ketrue=util.get_ke(self.ds[self.i],self.grid)
        _,self.kepred=util.get_ke(self.pred,self.grid)
        
        return
    
    def animate(self):
        fig, axs = plt.subplots(1, 5,figsize=(17,3))
        self.ax1=axs[0].imshow(self.ds[0], cmap=sns.cm.icefire,interpolation='none')
        fig.colorbar(self.ax1, ax=axs[0])
        axs[0].set_xticks([]); axs[0].set_yticks([])
        axs[0].set_title("Simulation")

        self.ax2=axs[1].imshow(self.ds[0], cmap=sns.cm.icefire,interpolation='none')
        fig.colorbar(self.ax2, ax=axs[1])
        axs[1].set_xticks([]); axs[1].set_yticks([])
        axs[1].set_title("Emulator")

        self.ax3=axs[2].imshow(self.ds[0], cmap=sns.cm.icefire,interpolation='none')
        fig.colorbar(self.ax3, ax=axs[2])
        axs[2].set_xticks([]); axs[2].set_yticks([])
        axs[2].set_title("Residuals")

        fig.tight_layout()
        
        ## Time evol metrics
        axs[3].set_title("Correlation")
        self.ax4=[axs[3].plot(-1),axs[3].plot(-1)]
        axs[3].set_ylim(0,1)
        axs[3].set_xlim(0,self.times[-1])

        axs[4].set_title("KE spectra")
        axs[4].set_ylim(1e1,1e7)
        axs[4].set_xlim(6e-1,3.5e1)
        self.ax5=[axs[4].loglog(-1),axs[4].loglog(-1)]
        self.ax5[0][0].set_xdata(self.k1d_plot)
        self.ax5[0][0].set_ydata(self.ketrue)
        
        self.ax5[1][0].set_xdata(self.k1d_plot)
        self.ax5[1][0].set_ydata(self.kepred)
        
        self.time_text=axs[2].text(-20,-20,"")
        
        fig.tight_layout()
        
        anim = animation.FuncAnimation(
                                       fig, 
                                       self.animate_func, 
                                       frames = self.nFrames,
                                       interval = 1000 / self.fps, # in ms
                                       )
        plt.close()
        
        if self.savestring:
            print("saving")
            # saving to m4 using ffmpeg writer 
            writervideo = animation.FFMpegWriter(fps=self.fps) 
            anim.save('%s.mp4' % self.savestring, writer=writervideo) 
            plt.close()
        else:
            return HTML(anim.to_html5_video())
        
    def animate_func(self,i):
        if i % self.fps == 0:
            print( '.', end ='' )
            
        self.i=i
        self.time_text.set_text("%d timesteps" % (i))
    
        ## Set image and colorbar for each panel
        image=self.ds[self.i].numpy()
        self.ax1.set_array(image)
        self.ax1.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
        
        image=self.pred.numpy()
        self.ax2.set_array(image)
        self.ax2.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
        
        image=(self.pred-self.ds[self.i]).numpy()
        self.ax3.set_array(image)
        self.ax3.set_clim(-np.max(np.abs(image)), np.max(np.abs(image)))
 
        self.ax4[0][0].set_xdata(np.array(self.times[0:len(self.correlation)]))
        self.ax4[0][0].set_ydata(np.array(self.correlation))
        
        self.ax4[1][0].set_xdata(np.array(self.times[0:len(self.autocorrelation)]))
        self.ax4[1][0].set_ydata(np.array(self.autocorrelation))

        self.ax5[0][0].set_xdata(self.k1d_plot)
        self.ax5[0][0].set_ydata(self.ketrue)
        
        self.ax5[1][0].set_xdata(self.k1d_plot)
        self.ax5[1][0].set_ydata(self.kepred)
        
        self._push_forward()
        
        return 
        

def long_run_figures(model,emu_run,steps=int(1e6),residual=True,sigma=None,exit_criterion=None):
    """ For a given emulator model and set of test ICs, run for `steps` iterations
        We will plot enstrophy over time, and plot the fields at the end of the rollout
        Here we are testing long term stability - on timescales longer than we can store
        test data for.

        model:    torch.nn.Module emulator
        emu_run:  initial conditions for all trajectories
        steps:    total number of steps to run
        residual: bool determining whether this is a state or residual emulator
        sigmas:   None for a deterministic trajectory, otherwise set the std dev value
                  for noise added at each timestep
        exit_criterion:
                  maximum enstrophy value at which the trajectory exits (prevents us from
                  running long trajectories after a trajectory has gone unstable)
        
        Test set will be just the initial states, i.e. [batch size, nx, ny]
        Also assuming model is already on GPU"""

    assert 0.9<emu_run.std().item()<1.1, "Fields are not normalised"

    ## Just going to assume we have a GPU available, as we are not doing long
    ## rollouts on CPU
    emu_run=emu_run.to("cuda")
    model=model.to("cuda")
    
    enstrophies=torch.zeros((len(emu_run),steps))
    if residual==True:
        with torch.no_grad():
            for aa in range(steps):
                emu_run=model(emu_run.unsqueeze(1)).squeeze()+emu_run
                enstrophies[:,aa]=abs(emu_run**2).sum(axis=(1,2))
    else:
        with torch.no_grad():
            for aa in range(steps):
                emu_run=model(emu_run.unsqueeze(1)).squeeze()
                enstrophies[:,aa]=abs(emu_run**2).sum(axis=(1,2))

    enstrophy_figure=plt.figure()
    plt.title("Enstrophy from long emulator rollout")
    for aa in range(len(enstrophies)):
        plt.plot(enstrophies[aa],color="gray",alpha=0.4)
    enstrophies=enstrophies[:aa]
    plt.ylim(1000,10000)
    plt.xlabel("Emulator timestep")
    plt.ylabel("Enstrophy")
   
    field_figure=plt.figure(figsize=(14,6))
    plt.suptitle("Model predictions after %d emulator steps" % steps)
    for aa in range(1,9):
        plt.subplot(2,4,aa)
        plt.imshow(emu_run[aa].cpu().squeeze(),cmap=sns.cm.icefire)
        plt.colorbar()
    return enstrophy_figure, field_figure

