#!/bin/bash
#SBATCH -p production
#SBATCH --time=2-00:00:00
#SBATCH --gres gpu:1
#SBATCH --mem-per-cpu=64G
#SBATCH --job-name=dna_llm
#SBATCH -o logs/%x-%N-%j.out
#SBATCH -e logs/%x-%N-%j.out

nvidia-smi
# might need to copy original HOME to a location slurm can access
#export HOME="path_to_home"

# set vars
NUM_GPUS=2
REPO_DIR="path_to/learning-nucleotides/"
EXPT_DIR="/$REPO_DIR/experiment_one"
CONFIG_FILE="/$EXPT_DIR/configs/$1"
cd $EXPT_DIR

source ../.venv/bin/activate
srun torchrun --nproc_per_node=$NUM_GPUS run_hf_model.py $CONFIG_FILE
echo "Deactivating virtual environment..."
deactivate