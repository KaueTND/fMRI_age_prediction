#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=170G
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a100

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!

# change permissions on slurm job output
((array_id=${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}))
# activating conda
#conda init bash
conda activate py_36

# Check for the input file
((i=SLURM_ARRAY_TASK_ID-1))

python eval_fmri_model.py

