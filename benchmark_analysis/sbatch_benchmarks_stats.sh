#!/bin/bash

#SBATCH --job-name=collect_stats
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100_7g.80gb:1
#SBATCH --time=10000
#SBATCH --mail-type=ALL
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module purge                        # Environment cleanup
module load python/anaconda3        # Loading of anaconda3 module
eval "$(conda shell.bash hook)"     # Shell initialization to use conda
conda activate sudowoodo_env                # Activation your python environment
python benchmark_properties.py join_benchmarks                    

