from py_module_Agent_observation_capabilities import map_bar_heights_to_xlabels_from_jsons

mapping = map_bar_heights_to_xlabels_from_jsons(
    "dir_demo_bar_chart_full_pipeline/ocr_data.json",
    "dir_demo_bar_chart_full_pipeline/inferred_axes_and_bars.json"
)

print(mapping)
