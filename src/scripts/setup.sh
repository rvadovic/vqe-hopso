#!/bin/bash

# initialize conda
source ~/miniconda3/etc/profile.d/conda.sh

#activate environment "vqe"
conda activate vqe

# setup OpenMPI paths
export PATH=/usr/lib64/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:$LD_LIBRARY_PATH