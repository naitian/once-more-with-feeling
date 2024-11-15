#!/bin/bash

for i in $(find data/splits/ -maxdepth 1 -type f -exec basename {} \;)
do
	SPLIT_NAME=$i sbatch segment.slurm;
done
