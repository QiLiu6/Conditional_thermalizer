#!/bin/bash

#SBATCH --job-name=Train_Emu
#SBATCH --time=24:00:00
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
	    --overlay /scratch/ql2221/Qi_singularity/overlay-50G-10M.ext3:ro \
	    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
	    /bin/bash -c "source /ext3/env.sh; python3 /home/ql2221/Projects/Conditional_thermalizer/Training/Train_emulator_config.py"
