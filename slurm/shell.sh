#!/bin/bash

# export APPTAINERENV_META_PATH=/app/src/video/metadata/secure_test_metadata.tsv
apptainer shell --nv --no-mount /etc/localtime --bind /path/to/movies/ ../container-latest.sif
