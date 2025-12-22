#!/usr/bin/env bash

set -e

ENV_NAME="my_aif_in_vis"
OUTPUT_FILE="my_aif_in_vis_env.yml"

echo "Saving conda environment: $ENV_NAME"
conda env export -n "$ENV_NAME" --no-builds > "$OUTPUT_FILE"

echo "Environment saved to $OUTPUT_FILE"

