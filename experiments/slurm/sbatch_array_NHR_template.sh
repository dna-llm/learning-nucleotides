#!/bin/bash

#SBATCH -A project_name
#SBATCH --job-name 2d_array
#SBATCH --partition <partition>
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user your@email.com

#SBATCH -o slurm_out/2d_array.%A_%a.out
#SBATCH -e slurm_out/2d_array.%A_%a.err

#SBATCH --time=01:00:00
#SBATCH --array=0-6

#SBATCH -G A100:1

eval "$(conda shell.bash hook)"
conda activate nucleotides

# token
python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')"

# Array of config files
CONFIGS=(
    experiments/configs/wavelet/wavelet-14m-2048-two_d.yaml
    experiments/configs/denseformer/denseformer_2048/denseformer-160m-2048-two_d.yaml
    experiments/configs/denseformer/denseformer_2048/denseformer-410m-2048-two_d.yaml
    experiments/configs/pythia/pythia_2048/pythia-410m-2028-two_d.yaml
    experiments/configs/pythia/pythia_2048/pythia-160m-2028-two_d.yaml
    experiments/configs/evo/evo_2048/evo-410m-2048-two_d.yaml
    experiments/configs/evo/evo_2048/evo-160m-2048-two_d.yaml
)

# Get the current array task ID
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

srun python experiments/run_hf_model.py $CONFIG