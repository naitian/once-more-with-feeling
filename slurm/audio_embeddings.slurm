#!/bin/bash

#SBATCH --job-name=audio_embeddings
#SBATCH --partition=srdc_GTX1080TI
#SBATCH --account=isch-aux01-access

#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

#SBATCH --gres=gpu:1

#SBATCH --output=logs/s-%x.%A_%a.out
#SBATCH --error=logs/s-%x.%A_%a.err

SPLIT_NAME=$(printf "%04d" $SLURM_ARRAY_TASK_ID)
echo $SPLIT_NAME

# export APPTAINERENV_META_PATH=/app/src/video/metadata/secure_test_metadata.tsv
# we add a 5-second offset between jobs to allow for free GPU detection to work better maybe?
cat data/imdb_splits/$SPLIT_NAME | parallel --delay 5 --joblog logs/parallel_joblog_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log -j 4 \
	'apptainer exec --nv --no-mount /etc/localtime --bind /path/to/movies/ ../container-latest.sif python src/audio/embeddings.py --audio_path=data/asr_extracts/audio/{}/ --output_path=data/asr_extracts/audio_embeddings/{}/{}.pkl' 2>&1 | tee -a logs/s-${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out

parallel_exit_status=${PIPESTATUS[1]}
exit $parallel_exit_status
