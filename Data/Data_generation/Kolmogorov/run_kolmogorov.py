import Data.Data_generation.Kolmogorov.simulate as simulate

import torch
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=str, help="name of config file")
parser.add_argument("--save_path", required=True, type=str, help="save_path")

args = parser.parse_args()

config = yaml.safe_load(Path(args.config).read_text())
config["save_path"] = args.save_path
config["viscosity"] = 1/config["reynolds"]
config["Dt/dt"] = config["Dt"] / config["dt"]
sim_stack=torch.tensor([],dtype=torch.float32)

for aa in tqdm(range(config["n_sims"])):
    sim = simulate.run_kolmogorov_sim(config["nsteps"], config["dt"], config["Dt"], spinup=config["spinup"], decorr_steps = config["decorr_steps"],
                                      vis=config["viscosity"], gridsize=config["gridsize"], downsample=config["downsample"], 
                                      n_traj = config["n_traj"], chunk_size=config["chunk_size"])
    sim = torch.tensor(sim.values, dtype=torch.float32)
    sim=sim.reshape(1,int(config["nsteps"],sim.shape[-1],sim.shape[-1]))
    sim_stack=torch.cat((sim_stack,sim.clone().detach()))

save_dict={"data_config":config,
           "data":sim_stack}

torch.save(save_dict, config["save_path"])
