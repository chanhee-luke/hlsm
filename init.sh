#!/bin/bash
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change this to your desired directory:
export WS_DIR=/local/scratch/song.1855/hlsm/workspace

# Stuff that ALFRED needs
export ALFRED_PARENT_DIR=${WS_DIR}/alfred_src
export ALFRED_ROOT=${ALFRED_PARENT_DIR}/alfred

export PYTHONPATH=$PYTHONPATH:${ALFRED_PARENT_DIR}:${ALFRED_ROOT}:${ALFRED_ROOT}/gen

# Stuff that HLSM needs
export LGP_WS_DIR="${WS_DIR}"
export LGP_MODEL_DIR="${WS_DIR}/models"
export LGP_DATA_DIR="${WS_DIR}/data"

# TEACH specific
export ET_DATA=/research/nfs_su_809/vln/teach-dataset
