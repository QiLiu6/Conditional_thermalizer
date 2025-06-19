import os
import wandb

import Training.Train_conditional_thermalizer_algo as Train_CT
import Models.misc as misc


## Stop jax hoovering up GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

config={}
config["input_channels"]=2
config["output_channels"]=1
config["model_type"]="ModernUnet"
config["dim_mults"]=[2,2,2] 
config["hidden_channels"]=64
config["activation"]="gelu"
config["loader_workers"]=1
config["image_size"]=64

#how many steps do we allow ourselves to look to the past
config["timesteps"] = 1000
config["lagsteps"] = 128

config["lag_embedding"] = 512
config["noise_timestep_embedding"] = 512

config["project"]="thermalizer"

if len(sys.argv) > 1:
    wandb_run_name = sys.argv[1]  # take the first argument
else:
    wandb_run_name = misc.rs()
config["wandb_run_name"] = wandb_run_name
config["norm"]=False
config["ddp"]=False
config["PDE"]="Kolmogorov"
config["file_path"]="/scratch/ql2221/thermalizer_data/kolmogorov/reynold10k/combined_data.p"
config["save_path"]="/scratch/ql2221/thermalizer_data/wandb_data"
config["subsample"]=None
config["train_ratio"]=0.95
config["save_name"]="model_weights.pt"

config["optimization"]={}
config["optimization"]["epochs"]=200
config["optimization"]["lr"]=0.0002
config["optimization"]["wd"]=0.05
config["optimization"]["batch_size"]=64
config["optimization"]["gradient_clipping"]=1.
config["optimization"]["scheduler_step"]=100000
config["optimization"]["scheduler_gamma"]=0.5

#if training from checkpoint uncomment this
# checkpoint_string = "/scratch/ql2221/thermalizer_data/wandb_data/wandb/run-20250521_134541-z6d1vc7f/files/checkpoint_last.p"
# trainer = Train_Unet.trainer_from_checkpoint(checkpoint_string)
# trainer.config["optimization"]["epochs"]= config["optimization"]["epochs"]

trainer = Train_CT.CTTrainer(config)
print(trainer.config["cnn learnable parameters"])
trainer.run()
