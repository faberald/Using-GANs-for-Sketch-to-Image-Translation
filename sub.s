#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --mem=5GB
#SBATCH --job-name=pix2pix
#SBATCH --mail-type=END
#SBATCH --mail-user=hl3635@nyu.edu
#SBATCH --output=slurm_%j.out


. ~/.bashrc
export PATH=$HOME/anaconda3/bin:$PATH
conda activate hl-pyt
module load cudnn/9.0v7.3.0.29
module load cuda/9.0.176

cd models/pix2pix
python pix2pix.py --n_epochs=20 --batch_size=64 --n_cpu=12