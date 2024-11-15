#!/bin/bash

#SBATCH --job-name=containertest
#SBATCH --partition=srdc
#SBATCH --account=isch-aux01-access

#SBATCH --time=00:00:30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

#SBATCH --output=logs/s-%x.%j.out
#SBATCH --error=logs/s-%x.%j.err

apptainer exec  --no-mount /etc/localtime --bind /path/to/movies/ ../container-latest.sif python scripts/list_all_imdb.py
