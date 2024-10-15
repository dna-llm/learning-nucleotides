#!/bin/bash

MODEL_DIRS="models_to_upload.txt"

while IFS= read -r model_dir || [[ -n "$model_dir" ]]; do
    # Skip lines starting with '#' or empty lines
    [[ "$model_dir" =~ ^# ]] || [[ -z "$model_dir" ]] && continue
    job_name=$(basename "$model_dir")
    echo "Submitting job for: $job_name"
    sbatch --job-name="upload-$job_name" hf_model_upload.slurm "$(realpath "$model_dir")"
done < "$MODEL_DIRS"

echo "All jobs have been submitted."