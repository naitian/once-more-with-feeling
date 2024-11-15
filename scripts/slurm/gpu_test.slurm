#!/bin/bash

#SBATCH --job-name=gputest
#SBATCH --partition=srdc_GTX1080TI
#SBATCH --account=isch-aux01-access

#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

#SBATCH --gres=gpu:1

# set up output file
#SBATCH --output=logs/s-%x.%j.out
#SBATCH --error=logs/s-%x.%j.err

nvidia-smi