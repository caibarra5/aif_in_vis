#!/usr/bin/env bash

SRC_DIRS=(
  dir_demo_bar_chart_full_pipeline_ok
  dir_demo_bar_chart_full_pipeline_partial
  dir_demo_bar_chart_full_pipeline_fail
)

DEST_DIR=dir_chart_structure_quality_summary

for d in "${SRC_DIRS[@]}"; do
  if [[ -f "$d/chart_structure_quality.json" ]]; then
    cp "$d/chart_structure_quality.json" \
       "$DEST_DIR/chart_structure_quality_$(basename "$d").json"
  else
    echo "WARNING: $d/chart_structure_quality.json not found"
  fi
done

