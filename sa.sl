#!/bin/bash

## This is an example of an sbatch script to run a pytorch script
## using Singularity to run the pytorch image.
##
## Set the DATA_PATH to the directory you want the job to run in.
##
## On the singularity command line, replace ./test.py with your program
##
## Change reserved resources as needed for your job.
##

#SBATCH --job-name=spec2cnn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH --partition=volta-gpu
#SBATCH --output=run-%j.log
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access
#SBATCH --mail-user=ross004@unc.edu
#SBATCH --mail-type=ALL

unset OMP_NUM_THREADS

# Set SIMG path
SIMG_PATH=/nas/longleaf/apps/pytorch_py3/1.9.1/simg

# Set SIMG name
SIMG_NAME=pytorch1.9.1-py3-cuda11.1-ubuntu18.04.simg

# Set data path
DATA_PATH=/work/users/r/o/ross004/560/model/

# GPU with Singularity
singularity exec --nv -B /work -B /proj $SIMG_PATH/$SIMG_NAME bash -c "cd $DATA_PATH; python ./sa.py"

# Retrieve Elapsed Time After Completion
ELAPSED_TIME=$(squeue -j "$JOB_ID" -o " %E")
echo "Job $JOB_ID finished at $(date)"
echo "Total runtime: $ELAPSED_TIME"
