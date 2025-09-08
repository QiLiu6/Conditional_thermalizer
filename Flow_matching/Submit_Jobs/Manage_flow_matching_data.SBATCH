#!/bin/bash

#SBATCH --job-name=Flow_Data
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=100GB
#SBATCH --output=slurm_%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=ql2221@nyu.edu

module purge

singularity exec --nv \
  --overlay /scratch/ql2221/Qi_singularity/overlay-50G-10M.ext3:ro \
  /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; \
  python3 /home/ql2221/Projects/Conditional_thermalizer/Flow_matching/Data/Manage_data.py \
  /scratch/ql2221/thermalizer_data/kolmogorov/reynold10k/Therm_rollout.p \
  /scratch/ql2221/thermalizer_data/kolmogorov/reynold10k/combined_data.p \
  /scratch/ql2221/thermalizer_data/kolmogorov/reynold10k/Flow_training_data.p"
