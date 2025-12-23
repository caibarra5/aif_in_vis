# demo_f11_global_coords.py

from py_module_Agent_observation_capabilities import (
    detect_primitives_text_aware_global_coords
)

result = detect_primitives_text_aware_global_coords(
    image_path="bar_graph_example_5bars.png",
    output_dir="demo_f11_text_aware_global",
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


import json
from pathlib import Path
import cv2

from py_module_Agent_observation_capabilities import (
    detect_primitives_text_aware_global_coords
)

from py_module_Agent_observation_capabilities import (
    infer_axes_and_bars_from_primitives,
    annotate_axes_and_bars,
    save_inference_json
)

# --------------------------------------------------
# Existing call (unchanged)
# --------------------------------------------------
result = detect_primitives_text_aware_global_coords(
    image_path="bar_graph_example_5bars.png",
    output_dir="demo_f11_text_aware_global",
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

# --------------------------------------------------
# NEW: infer bars + axes and annotate
# --------------------------------------------------
out_dir = Path("demo_f11_text_aware_global")

with open(out_dir / "primitives.json", "r", encoding="utf-8") as f:
    primitives = json.load(f)

img = cv2.imread("bar_graph_example_5bars.png")
H, W = img.shape[:2]

inference = infer_axes_and_bars_from_primitives(
    primitives,
    image_width=W,
    image_height=H
)

save_inference_json(
    inference,
    out_dir / "inferred_axes_and_bars.json"
)

annotate_axes_and_bars(
    image_path="bar_graph_example_5bars.png",
    inference=inference,
    output_path=out_dir / "annotated_axes_and_bars.png"
)
