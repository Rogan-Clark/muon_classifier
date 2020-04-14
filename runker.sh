#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=htc
#SBATCH --time=120:00:00
#SBATCH --mem=12288
#SBATCH --job-name=pointrun_allparams
#SBATCH --gres=gpu:k80:1
#SBATCH -error /home/exet4487/logs/pointrun_allparams.out
#SBATCH -output /home/exet4487/logs/pointrun_allparams.err

module load python/anaconda3/5.0.1
module load gpu/cuda/10.0.130
module load gpu/cudnn/7.5.0__cuda-10.0

source activate $HOME/k2
python paramlstm.py
