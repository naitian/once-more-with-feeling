#!/bin/bash

# Uncomment this line to use the shared test videos
# export APPTAINER_META_PATH=/app/src/video/metadata/secure_test_metadata.tsv

mkdir -p data/
apptainer exec --no-mount /etc/localtime \
	../container-latest.sif \
	python /app/scripts/list_all_imdb.py > data/all_imdb.txt

mkdir -p data/imdb_splits
rm -rf data/imdb_splits/*
split -a 4 -d -l 24 data/all_imdb.txt data/imdb_splits/


