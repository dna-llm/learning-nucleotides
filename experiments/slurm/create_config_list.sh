#!/bin/bash

CONFIGS_DIR="../configs/$1"
OUTPUT_FILE="config_files_to_run.txt"

# Remove the output file if it already exists
if [ -f "$OUTPUT_FILE" ]; then
    echo "Removing existing output file..."
    rm "$OUTPUT_FILE"
fi

# Loop through each yaml file in the configs folder
for file in "$CONFIGS_DIR"/*.yaml; do
    RELATIVE_PATH=$(realpath --relative-to="../configs" "$file")
    echo "$RELATIVE_PATH" >> "$OUTPUT_FILE"
done
