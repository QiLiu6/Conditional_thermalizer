#!/bin/bash

#SBATCH --job-name=therm_kol
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|v100"
#SBATCH --mem=128GB
#SBATCH --output=slurm_%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=ql2221@nyu.edu

module purge

singularity exec --nv \
	    --overlay /scratch/ql2221/torch_sing/overlay-15GB-500K.ext3:ro \
	    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
	    /bin/bash -c "source /ext3/env.sh; python3 /home/ql2221/Projects/Conditional_thermalizer/CT/Data/Data_generation/Kolmogorov_thermalized/Thermalized_kolmogorov.py
"
