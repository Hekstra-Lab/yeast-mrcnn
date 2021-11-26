#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 120
#SBATCH -p seas_gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --mem 48G
#SBATCH --constraint v100
#SBATCH -o output.txt
#SBATCH -e errors.txt

module load cudnn/8.1.0.77_cuda11.2-fasrc01
python train_bbbc.py /n/holyscratch01/hekstra_lab/russell/bbbc_nuclei ~/microscopy-notebooks/yeast_mrcnn_train cuda
