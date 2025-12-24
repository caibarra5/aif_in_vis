from py_module_Agent_observation_capabilities import (
    run_bar_chart_full_pipeline
)

image_path = "bar_graph_example_5bars.png"
output_dir = "dir_demo_bar_chart_full_pipeline"

result = run_bar_chart_full_pipeline(
    image_path=image_path,
    output_dir=output_dir,
    primitives_kwargs=dict(
        min_line_length=60,
        hough_threshold=80,
        angle_tol_deg=5.0
    ),
    ocr_kwargs=dict(
        psm=6,
        min_confidence=30.0
    )
)

print("Done.")
for k, v in result.items():
    print(f"{k}: {v}")
