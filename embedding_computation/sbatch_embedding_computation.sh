#!/bin/bash
#SBATCH --job-name=hytrel_inference
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_7g.80gb:1
#SBATCH --time=30000
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
module purge                        # Environment cleanup
# module load python/anaconda3 # Loading of anaconda3 module
# module load cuda/11.1 ##loading cuda module 
module load gcc/10 ##loading gcc compiler 
eval "$(conda shell.bash hook)"     # Shell initialization to use conda
conda activate hytrel_env                # Activation your python environment
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python compute_embeddings.py