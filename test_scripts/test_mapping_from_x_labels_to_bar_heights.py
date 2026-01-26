from py_module_Agent_observation_capabilities import (
    map_bar_heights_to_xlabels_from_jsons
)

from py_module_Agent_aif_capabilities import (
    summarize_bar_chart_to_json
)

# Step 1: extract bar chart as dictionary
mapping = map_bar_heights_to_xlabels_from_jsons(
    "dir_demo_bar_chart_full_pipeline/ocr_data.json",
    "dir_demo_bar_chart_full_pipeline/inferred_axes_and_bars.json"
)

# Step 2: summarize + save JSON
summary = summarize_bar_chart_to_json(
    bar_dict=mapping,
    output_dir="./dir_demo_bar_chart_full_pipeline",
    filename="bar_chart_summary.json"
)
