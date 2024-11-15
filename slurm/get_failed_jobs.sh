#!/bin/bash


JOBID=$1

tail -q -n+2 logs/parallel_joblog_${JOBID}_* | awk '$7 == 1'
