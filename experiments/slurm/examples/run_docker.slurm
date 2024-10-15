#!/bin/bash
#SBATCH -p production
#SBATCH --time=5-00:00:00
#SBATCH --gres gpu:1
#SBATCH --mem-per-cpu=64G
#SBATCH --job-name=dna_llm
#SBATCH -o logs/%x-%N-%j.out
#SBATCH -e logs/%x-%N-%j.out


# set HF_API_TOKEN
REPO_DIR="path_to/learning-nucleotides"
EXPT_DIR="/$REPO_DIR/experiments"
CONFIG_FILE="/$EXPT_DIR/configs/$1"
export HF_API_TOKEN="hf_..."
# load and build docker image
cd $REPO_DIR
docker build -f $REPO_DIR/docker/experiment_one/Dockerfile -t training_image .
# run the docker container
docker run --gpus all \
    -e HF_API_TOKEN=$HF_API_TOKEN \
    -v $CONFIG_FILE:/app/config.yaml \
    -f /app/config.yaml \
    training_image