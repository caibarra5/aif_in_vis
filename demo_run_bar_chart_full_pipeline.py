from py_module_Agent_observation_capabilities import (
    run_bar_chart_full_pipeline
)

image_path = "bar_graph_example_5bars.png"

# --------------------------------------------------
# CASE 1: Baseline (works)
# --------------------------------------------------
result_ok = run_bar_chart_full_pipeline(
    image_path=image_path,
    output_dir="dir_demo_bar_chart_full_pipeline_ok",
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

print("\n=== BASELINE ===")
for k, v in result_ok.items():
    print(f"{k}: {v}")

# --------------------------------------------------
# CASE 2: Partial failure (miss some text / axes)
# --------------------------------------------------
result_partial = run_bar_chart_full_pipeline(
    image_path=image_path,
    output_dir="dir_demo_bar_chart_full_pipeline_partial",
    primitives_kwargs=dict(
        min_line_length=120,      # too strict → miss axes
        hough_threshold=120,      # fewer lines
        angle_tol_deg=2.0         # too strict on orientation
    ),
    ocr_kwargs=dict(
        psm=11,                   # sparse text mode
        min_confidence=60.0       # discard many labels
    )
)

print("\n=== PARTIAL FAILURE ===")
for k, v in result_partial.items():
    print(f"{k}: {v}")

# --------------------------------------------------
# CASE 3: Near-total failure (almost nothing detected)
# --------------------------------------------------
result_fail = run_bar_chart_full_pipeline(
    image_path=image_path,
    output_dir="dir_demo_bar_chart_full_pipeline_fail",
    primitives_kwargs=dict(
        min_line_length=200,      # kill axes
        hough_threshold=200,
        angle_tol_deg=1.0
    ),
    ocr_kwargs=dict(
        psm=11,
        min_confidence=85.0       # kill OCR almost entirely
    )
)

print("\n=== FAILURE CASE ===")
for k, v in result_fail.items():
    print(f"{k}: {v}")
