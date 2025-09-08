import os
import pickle
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cmocean
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import sys

import Flow_matching.Models.Unet as Unet
import Flow_matching.Models.misc as misc

import Flow_matching.Data.Dataset as datasets

def setup():
    """Sets up the process group for distributed training.
       We are using torchrun so not using rank and world size arguments """
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")

def cleanup():
    """Cleans up the process group."""
    dist.destroy_process_group()

class Trainer:
    """ Base trainer class """
    def __init__(self,config):
        self.config  =config
        self.epoch = 1 ## Initialise at first epoch
        self.training_step = 0 ## Counter to keep track of number of weight updates
        self.wandb_init = False ## Bool to track whether or not wandb run has been initialised
        self.ema=None
        if self.config.get("wandb_log_freq"):
            self.log_freq = self.config.get("wandb_log_freq")
        else:
            self.log_freq = 50

        self.gradient_clip = self.config["optimization"].get("gradient_clip")

        '''
        Don't know what this part does.
        '''
        if self.config["ddp"]:
            setup()
            self.gpu_id=int(os.environ["LOCAL_RANK"])
            self.ddp=True
            self.world_size=dist.get_world_size()
            self.config["world_size"]=self.world_size
            if self.gpu_id==0:
                self.logging=True
            else:
                self.logging=False
        else:
            self.gpu_id="cuda"
            self.ddp=False
            self.logging=True

        print("Prep data")
        self._prep_data()
        print("Prep model")
        self._prep_model()
        print("Prep optimizer")
        self._prep_optimizer()

    #I also don't know how the wandb stuff work
    def init_wandb(self):
        ## Set up wandb stuff
        wandb.init(entity="qiliu2221",
                   project=self.config["project"],
                   dir="/scratch/ql2221/thermalizer_data/wandb_data",
                   name=self.config["wandb_run_name"],
                   config=self.config,
                   )
        self.config["save_path"]=wandb.run.dir
        self.config["wandb_url"]=wandb.run.get_url()
        self.wandb_init=True 
        ## Sync all configs
        wandb.config.update(self.config, allow_val_change=True)
        self.model.config=self.config

    def resume_wandb(self):
        """ Resume a wandb run from the self.config wandb url. """
        wandb.init(entity="qiliu2221",project=self.config["project"],
                            id=self.config["wandb_url"][-8:],dir="/scratch/ql2221/thermalizer_data/wandb_data", resume="must")
        self.wandb_init=True
        return

    def _prep_data(self):
        #debugging
        if self.config["PDE"] == "Kolmogorov":
            train_data, valid_data, config = datasets.parse_data_file(self.config)
        else:
            print("Need to know what PDE system we are working with")
            quit()
    
        ds_train = datasets.FluidDataset(train_data)
        ds_valid = datasets.FluidDataset(valid_data)
        self.config = config ## Update config dict

        if self.ddp:
            train_sampler = DistributedSampler(ds_train)
            valid_sampler = DistributedSampler(ds_valid)
        else:
            train_sampler=RandomSampler(ds_train)
            valid_sampler=RandomSampler(ds_valid)
    
        self.train_loader = DataLoader(
                ds_train,
                num_workers = self.config["loader_workers"],
                batch_size = self.config["optimization"]["batch_size"],
                sampler = train_sampler,
            )
    
        self.valid_loader = DataLoader(
                ds_valid,
                num_workers = self.config["loader_workers"],
                batch_size = self.config["optimization"]["batch_size"],
                sampler = valid_sampler,
            )

    def _prep_model(self):
        self.model=misc.model_factory(self.config).to(self.gpu_id)
        self.config["cnn learnable parameters"]=sum(p.numel() for p in self.model.parameters())

        if self.config.get("ema_decay"):
            self.ema=misc.ExponentialMovingAverage(self.model,decay=self.config.get("ema_decay"))
            self.ema.register()

        if self.ddp:
            self.model = DDP(self.model,device_ids=[self.gpu_id])

    def _prep_optimizer(self):
        self.criterion=nn.MSELoss()
        self.optimizer=torch.optim.AdamW(self.model.parameters(),
                            lr=self.config["optimization"]["lr"],
                            weight_decay=self.config["optimization"]["wd"])
        if self.config["optimization"].get("scheduler_step"):
            self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,
                    self.config["optimization"]["scheduler_step"],
                    gamma=self.config["optimization"]["scheduler_gamma"], last_epoch=-1)
        else:
            self.scheduler=None

    def training_loop(self):
        raise NotImplementedError("Implemented by subclass")

    def valid_loop(self):
        raise NotImplementedError("Implemented by subclass")

    def run(self):
        raise NotImplementedError("Implemented by subclass")

class Flow_matching_Trainer(Trainer):
    def __init__(self,config):
        super().__init__(config)
        if self.config["ddp"]==True:
            raise NotImplementedError

    def load_checkpoint(self,file_string,resume_wandb=True):
        """ Load checkpoint from saved file """
        with open(file_string, 'rb') as fp:
            model_dict = pickle.load(fp)
        # model_dict["config"]["file_path"] = self.config["file_path"]
        # assert model_dict["config"]==self.config, "Configs not the same"
        self.model=misc.load_diffusion_model(file_string).to(self.gpu_id)
        self._prep_optimizer()
        self.optimizer.load_state_dict(model_dict['optimizer_state_dict'])
        self.epoch=model_dict["epoch"]
        self.training_step=model_dict["training_step"]

        if self.wandb_init==False and resume_wandb:
            self.resume_wandb()

        return

    def training_loop(self):
        """ Training loop for Unified Unet for learning conditional distribution"""
        self.model.train()
        for j,image in enumerate(self.train_loader):
            image = image.to(self.gpu_id)
            self.optimizer.zero_grad()
            
            t = torch.randint(1, self.config["rollout_steps"]-1, (image.shape[0],))

            # Create batch indices
            batch_indices = torch.arange(image.shape[0], device=image.device)
            
            # Advanced indexing to get different timesteps per batch item
            true_t = image[batch_indices, t, :, :].unsqueeze(1)      # Add channel dim back
            emu_t = image[batch_indices, 128+t, :, :].unsqueeze(1)   # Add channel dim back
            
            # Now true_t and emu_t have shape (batch_size, 1, 64, 64)

            s = torch.rand(image.shape[0]).to(image.device)
            # Stochastic interpolant - reshape s for broadcasting
            s_broadcast = s.view(-1, 1, 1, 1)  # Shape: (batch_size, 1, 1, 1)
            interpolant = s_broadcast * true_t + (1 - s_broadcast) * emu_t

            pred_diff = self.model(interpolant,t,s)
            true_diff = true_t - emu_t
            
            loss = self.criterion(pred_diff,true_diff)
            loss.backward()

            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            if self.logging and (self.training_step%self.log_freq==0):
                log_dic={}
                log_dic["train_loss"]=loss.item()
                log_dic["training_step"]=self.training_step
                if self.scheduler:
                    log_dic["lr"]=self.scheduler.get_last_lr()[-1]
                wandb.log(log_dic)
            self.training_step+=1

            if self.ema:
                self.ema.update()

        return loss

    def valid_loop(self):
        """ Training loop for UUnet. Aggregate loss over validation set for wandb update """
        log_dic = {}
        self.model.eval()
        if self.ema:
            self.ema.apply_shadow()
        epoch_loss = 0
        nsamp = 0
        with torch.no_grad():
            for x_valid in self.valid_loader:
                x_valid = x_valid.to(self.gpu_id)
                nsamp += x_valid.shape[0]
                loss = 0
                
                t = torch.randint(1, self.config["rollout_steps"]-1, (x_valid.shape[0],))
                batch_indices = torch.arange(x_valid.shape[0], device=x_valid.device)
                true_t = x_valid[batch_indices, t, :, :].unsqueeze(1)
                emu_t = x_valid[batch_indices, 128+t, :, :].unsqueeze(1)
                s = torch.rand(x_valid.shape[0]).to(x_valid.device)
                s_broadcast = s.view(-1, 1, 1, 1)
                interpolant = s_broadcast * true_t + (1 - s_broadcast) * emu_t
                pred_diff = self.model(interpolant,t,s)
                true_diff = true_t - emu_t
                loss = self.criterion(pred_diff,true_diff)

                epoch_loss+=loss.detach()*x_valid.shape[0]
               
        epoch_loss/=nsamp ## Average over full epoch
        ## Now we want to allreduce loss over all processes
        if self.ddp:
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
            ## Average across all processes
            self.val_loss = epoch_loss.item()/self.world_size
        else:
            self.val_loss = epoch_loss.item()
        if self.logging:
            log_dic={}
            log_dic["valid_loss"]=self.val_loss ## Average over full epoch
            log_dic["training_step"]=self.training_step
            wandb.log(log_dic)
        if self.ema:
            self.ema.restore()
        return loss

    def save_checkpoint(self, checkpoint_string):
        """ Checkpoint model and optimizer """

        save_dict={
                    'epoch': self.epoch,
                    'training_step': self.training_step,
                    'state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'config':self.config,
                    }
        with open(checkpoint_string, 'wb') as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def run(self,epochs=None):
        if self.logging and self.wandb_init==False:
            self.init_wandb()
            self.model.config=self.config ## Update Unet config too, missed in parent call

        if epochs:
            max_epochs=epochs
        else:
            max_epochs=self.config["optimization"]["epochs"]

        for epoch in range(self.epoch,max_epochs+1):
            self.epoch=epoch
            print("Training at epoch", self.epoch)
            self.training_loop()
            self.valid_loop()
            self.save_checkpoint(self.config["save_path"]+"/checkpoint_last.p")
            if self.ema:
                self.ema.apply_shadow()
                self.save_checkpoint(self.config["save_path"]+"/checkpoint_last_ema.p")
                self.ema.restore()
            
        print("DONE on rank", self.gpu_id)
        return

def trainer_from_checkpoint(checkpoint_string):
    with open(checkpoint_string, 'rb') as fp:
        model_dict = pickle.load(fp)
    if model_dict["config"]["model_type"]=='ModernUnet':
        trainer = UUnetTrainer(model_dict["config"])
    else:
        print("You are probably using Chris's code again")
        quit()
    trainer.load_checkpoint(checkpoint_string)
    return trainer
