from py_module_Agent_observation_capabilities import (
    chart_structure_quality_summary
)

base_dirs = [
    "dir_demo_bar_chart_full_pipeline_ok",
    "dir_demo_bar_chart_full_pipeline_partial",
    "dir_demo_bar_chart_full_pipeline_fail",
]

for d in base_dirs:
    print(f"\n=== Testing {d} ===")

    ocr_json = f"{d}/ocr_data.json"
    inference_json = f"{d}/inferred_axes_and_bars.json"
    output_json = f"{d}/chart_structure_quality.json"

    summary = chart_structure_quality_summary(
        ocr_json_path=ocr_json,
        inference_json_path=inference_json,
        output_json_path=output_json
    )

    for k, v in summary.items():
        print(f"{k}: {v}")
