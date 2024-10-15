#!/bin/bash

MODEL_DIR=$1
OUTPUT_FILE="models_to_upload.txt"

# Remove the output file if it already exists
if [ -f "$OUTPUT_FILE" ]; then
    echo "Removing existing output file..."
    rm "$OUTPUT_FILE"
fi
# Loop through each yml file in the configs folder
for file in "$MODEL_DIR/virus-"*; do
    echo "$file" >> "$OUTPUT_FILE"
done
