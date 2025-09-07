import torch.nn as nn
import torch
import math
from tqdm import tqdm
from scipy.stats import truncnorm
import torch
        
        
class Diffusion(nn.Module):
    def __init__(self,config,model,silence=True):
        """ 
        Constructor of the Diffusion model.

        Inputs: 
                Config: dictionary     : composed of all the configurations for the diffusion model
                model: torch.nn.Module : based architecture for the diffusion model (we use Unet)
                silence: boolean       : silence tqdm outputs
        """
        super().__init__()
        self.config=config
        ## Store number of timesteps
        if "timesteps" in self.config:
            self.timesteps = self.config["timesteps"]
        if "lagsteps" in self.config:
            self.lagsteps = self.config["lagsteps"]
        self.in_channels=self.config["input_channels"]
        self.image_size=self.config["image_size"]
        
        ## Check if we want to include timestep information
        if "lag_embedding" in self.config:
            self.lag_embedding = self.config["lag_embedding"]
        else:
            self.lag_embedding=None

        if "noise_timestep_embedding" in self.config:
            self.noise_timestep_embedding = self.config["noise_timestep_embedding"]
        else:
            self.noise_timestep_embedding = None

        self.silence=silence
        self.sampled_times=[]        

        betas=self._cosine_variance_schedule(self.timesteps)

        alphas=1.-betas
        alphas_cumprod=torch.cumprod(alphas,dim=-1)

        self.register_buffer("betas",betas)
        self.register_buffer("alphas",alphas)
        self.register_buffer("alphas_cumprod",alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod",torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",torch.sqrt(1.-alphas_cumprod))
        
        self.model=model

    def forward(self, x, noise, delta, predict_noise_level=False):
        '''
        All together method for taking a image, run the forward diffusion and let the model predict the noise

        Inputs:
                x: torch tensor (B x R x L x L) : need to include the snapshot x_t and x_{t-Delta}
                noise: torch tensor (same as x) : The noise to pass in for forward_diffusion
                delta: int
        Outputs:
                pred_noise: torch tensor (same as x) : the model's prediction of noise after seeing the noised image.
        '''
        delta = torch.full((x.shape[0],), delta, dtype=torch.int64, device=x.device)
        
        #generate random ts
        t = torch.randint(0,self.timesteps-1,(x.shape[0],)).to(x.device)
        self.sampled_times.append(t)

        # Input has size B x T x L x L, we only select the Tth snapshot
        x_t = x[:,-1:]
        
        #could be better, should just pass in noise with the right size
        noise_t = noise[:,-1:]
        x_t_noised = self._forward_diffusion(x_t,t,noise_t)

        #get the conditional frame
        x_t_minus = torch.zeros_like(x_t)
        for i in range(len(x)):
            x_t_minus[i] = x[i,-1-delta[i]:-delta[i],:,:]
                
        #concatenate the vectors so that they can be fed into the neural net
        x_noised_plus_x_t_minus = torch.cat((x_t_noised, x_t_minus), dim=1)
        pred_noise=self.model(x_noised_plus_x_t_minus, delta, t)
            
        return pred_noise

    @torch.no_grad()
    def sampling(self, n_samples, x_t_minus, delta, clipped_reverse_diffusion=None,device="cuda"):
        """ Generate fresh samples from pure noise conditioned on one snapshot:
        
        Inputs:
        n_samples: int                     : number of samples we want to generate
        x_t_minus: torch.tensor (1x1x64x64):
        clipped_reverse_diffusion: idk what this is doing
        
        Outputs:
        x_t: torch.tensor (n_samplesx1x64x64):  
        """
        x_t=torch.randn((n_samples,self.model.n_output_scalar_components,self.image_size,self.image_size)).to(device)
        original_noise = x_t.clone()
        x_conditional = torch.cat((x_t, x_t_minus), dim=1)

        for i in tqdm(range(self.timesteps-1,-1,-1),desc="Sampling"):
            noise=torch.randn_like(x_t).to(device)
            t=torch.tensor([i for _ in range(n_samples)]).to(device)
            x_t=self._reverse_diffusion(x_conditional,delta,t,noise)
            x_conditional = torch.cat((x_t, x_t_minus), dim=1)

        return x_t, original_noise

    @torch.no_grad()
    def denoising(self,x,delta,denoising_timestep,device="cuda",silence = False):
        """ Pass validation samples, x, and some denoising timestep.
            Add noise using forward diffusion, denoise these samples and return
            both the forward diffused and denoised images, after dewhitening if
            we are doing whitening """

        ## Noise timestep
        t=(torch.ones(len(x),dtype=torch.int64)*denoising_timestep).to(device)
        noise=torch.randn_like(x[:,-1:]).to(device)
        x_t = x[:,0:1]
        x_t_minus = x[:,1:2]
        x_t_noised = self._forward_diffusion(x_t,t,noise)
        x_noised_plus_x_t_minus = torch.cat((x_t_noised, x_t_minus), dim=1)
        x_t = x_t.to(device)
        x_conditional = x_noised_plus_x_t_minus.to(device)
        x_t_minus = x_t_minus.to(device)
        for i in tqdm(range(denoising_timestep-1,-1,-1),desc="Denoising"):
            noise=torch.randn_like(x_t).to(device)
            t=torch.tensor([i for _ in range(len(x))]).to(device)
            x_t=self._reverse_diffusion(x_conditional,delta,t,noise)
            x_conditional = torch.cat((x_t, x_t_minus), dim=1)
            
        return x_t

    def _cosine_variance_schedule(self,timesteps,epsilon= 0.008):
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*math.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)

        return betas

    def _forward_diffusion(self,x_0,t,noise=None):
        """ Run forward diffusion process, i.e. add noise to some input images
        x_0:    input tensors to add noise to
        t:      noise level to add. Can be either a tensor with same length x_0, in which case
                each image can be noised differently. Or just pass a scalar, and the same level of noise
                will be added to each image
        noise:  Tensor of random noise. Can be None, in which case we will generate noise here
        
        returns a tensor of the same shape x_0, where each image has been noised """

        ## If t is just an int, create a tensor for the forward process
        if type(t)==int:
            t=t*torch.ones(len(x_0),device=x_0.device,dtype=torch.int64)

        if noise==None:
            noise=torch.randn_like(x_0).to(x_0.device)

        assert x_0.shape==noise.shape
        #q(x_{t}|x_{0})
        return self.sqrt_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*x_0+ \
                self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*noise

    @torch.no_grad()
    def _reverse_diffusion(self,x_conditional,delta,t,noise):
        x_t = x_conditional[:,0:1]
        pred = self.model(x_conditional,delta,t)

        alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        sqrt_one_minus_alpha_cumprod_t=self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        mean=(1./torch.sqrt(alpha_t))*(x_t-((1.0-alpha_t)/sqrt_one_minus_alpha_cumprod_t)*pred)

        if t.min()>0:
            alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(x_t.shape[0],1,1,1)
            std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
        else:
            std=0.0
        return mean+std*noise 