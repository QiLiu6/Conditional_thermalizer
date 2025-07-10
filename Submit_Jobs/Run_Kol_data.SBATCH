#!/bin/bash

#SBATCH --job-name=run_kol_sims
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
	    --overlay /scratch/ql2221/Qi_singularity/overlay-50G-10M.ext3:ro \
	    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
	    /bin/bash -c "source /ext3/env.sh; python3 /home/ql2221/Projects/Conditional_thermalizer/Data/Data_generation/Kolmogorov/run_kolmogorov.py --config /home/ql2221/Projects/Conditional_thermalizer/Data/Data_Generation/Kolmogorov/data_config.yml --save_path /scratch/ql2221/thermalizer_data/kolmogorov/reynold10k/test.p
"
