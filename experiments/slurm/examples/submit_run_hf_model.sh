#!/bin/bash

CONFIG_FILES="config_files_to_run.txt"
# Submit all specified files
for file in $(cat $CONFIG_FILES); do
    # Skip lines starting with '#' or empty lines
    [[ "$file" =~ ^# ]] || [[ -z "$file" ]] && continue
    echo "$file submitted to slurm";
    JOB_NAME=$(basename "$file" .yaml)
    sbatch --job-name=$JOB_NAME run_hf_model.slurm $file

done