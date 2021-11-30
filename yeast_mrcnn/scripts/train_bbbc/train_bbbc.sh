#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 720
#SBATCH -p seas_gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --mem 48G
#SBATCH --constraint v100
#SBATCH -o output.txt
#SBATCH -e errors.txt

python train_bbbc.py /n/holyscratch01/hekstra_lab/russell/bbbc_nuclei/train/ ~/microscopy-notebooks/yeast_mrcnn_train/ cuda
