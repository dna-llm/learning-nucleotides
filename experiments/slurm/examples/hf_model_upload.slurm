#!/bin/bash
#SBATCH -p production
#SBATCH --time=2-00:00:00
#SBATCH --gres gpu:1
#SBATCH --mem-per-cpu=64G
#SBATCH --job-name=hf_upload
#SBATCH -o logs/%x-%N-%j.out
#SBATCH -e logs/%x-%N-%j.out

#export HOME="path_to_home"

# set vars
REPO_DIR="path_to/learning-nucleotides"
EXPT_DIR="/$REPO_DIR/experiment_one"
CONFIG_FILE="/$EXPT_DIR/configs/$1"
cd $EXPT_DIR
echo $1
MODEL_DIR="$EXPT_DIR/models_to_upload/$1"
echo "Uploading $MODEL_DIR to Hugging Face..."
source ../.venv/bin/activate
srun python -m utils.hf_upload $MODEL_DIR
echo "Deactivating virtual environment..."
deactivate
