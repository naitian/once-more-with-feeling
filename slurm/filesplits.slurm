#!/bin/bash

#SBATCH --job-name=chunk_files
#SBATCH --partition=srdc
#SBATCH --account=isch-aux01-access

#SBATCH --time=4:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

#SBATCH --output=logs/s-%x.%j.out
#SBATCH --error=logs/s-%x.%j.err


# Uncomment this line to use the shared test videos
# export APPTAINER_META_PATH=/app/src/video/metadata/secure_test_metadata.tsv

mkdir -p data/
apptainer -d exec --no-mount /etc/localtime \
	../container-latest.sif \
	python -c "from src.video.meta import VideoMetadata; print('\n'.join([str(m.path) for m in VideoMetadata.itermovies()]))" > data/paths.txt

mkdir -p data/splits
rm -rf data/splits/*
split -a 4 -d -l 24 data/paths.txt data/splits/


