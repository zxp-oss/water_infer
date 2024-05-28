#!/usr/bin/env bash

export LANG="en_US.UTF-8"
CURDIR=$(cd "$(dirname "$0")";pwd)

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
cd $CURDIR
. /mnt/geovis/iFactory6/plugin/anaconda3/etc/profile.d/conda.sh
conda activate torch
cd $CURDIR
python process_memory.py $1
conda deactivate
