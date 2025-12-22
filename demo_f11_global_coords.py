# demo_f11_global_coords.py

from py_module_Agent_observation_capabilities import (
    detect_primitives_text_aware_global_coords
)

result = detect_primitives_text_aware_global_coords(
    image_path="bar_graph_example_5bars.png",
    output_dir="demo_f11_text_aware_global",
    primitives_kwargs=dict(
        min_contour_area=200,
        min_bar_extent=0.60
    ),
    ocr_kwargs=dict(
        psm=6,
        min_confidence=30.0
    )
)

print("Done.")
for k, v in result.items():
    print(f"{k}: {v}")

