#!/bin/bash

#SBATCH --job-name=credits
#SBATCH --partition=srdc
#SBATCH --account=isch-aux01-access

#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

#SBATCH --output=logs/s-%x.%A_%a.out
#SBATCH --error=logs/s-%x.%A_%a.err

SPLIT_NAME=$(printf "%04d" $SLURM_ARRAY_TASK_ID)
echo $SPLIT_NAME

# export APPTAINERENV_META_PATH=/app/src/video/metadata/secure_test_metadata.tsv
# we add a 5-second offset between jobs to allow for free GPU detection to work better maybe?
cat data/splits/$SPLIT_NAME | parallel --joblog logs/parallel_joblog_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log -j 16 \
	'apptainer exec --no-mount /etc/localtime --bind /path/to/movies/,/path/to/shots/ ../container-latest.sif python scripts/data/find_credits.py {} --save_video' 2>&1 | tee -a logs/s-${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out

parallel_exit_status=${PIPESTATUS[1]}
exit $parallel_exit_status
