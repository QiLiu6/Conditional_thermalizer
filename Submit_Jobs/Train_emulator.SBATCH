#!/bin/bash

#SBATCH --job-name=Train_Emu
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --mem=128GB
#SBATCH --output=slurm_%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=ql2221@nyu.edu
#SBATCH --array=1

module purge

singularity exec --nv \
	    --overlay /scratch/ql2221/torch_sing/overlay-15GB-500K.ext3:ro \
	    /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
	    /bin/bash -c "source /ext3/env.sh; python3 /home/ql2221/Projects/Conditional_thermalizer/Training/Train_emulator_config.py Plain_emulator"
