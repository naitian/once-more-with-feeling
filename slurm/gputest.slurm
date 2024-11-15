#!/bin/bash

#SBATCH --job-name=gputest
#SBATCH --partition=srdc_GTX1080TI
#SBATCH --account=isch-aux01-access

#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

#SBATCH --gres=gpu:1

#SBATCH --output=logs/s-%x.%A_%a.out
#SBATCH --error=logs/s-%x.%A_%a.err

# export APPTAINERENV_META_PATH=/app/src/video/metadata/secure_test_metadata.tsv
apptainer exec --nv --no-mount /etc/localtime --bind /path/to/movies/ ../container-latest.sif nvidia-smi -q -d Memory | grep -A5 GPU | grep Free
