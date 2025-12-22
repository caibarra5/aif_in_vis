# demo_f6_f7_full_image.py

from pathlib import Path
import shutil
import json
import cv2

from py_module_Agent_observation_capabilities import (
    detect_chart_primitives,  # Function 6
    extract_chart_text_ocr,   # Function 7
)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
RUN_DIR = Path("./")
IMAGE_PATH = RUN_DIR / "bar_graph_example_5bars.png"

OUT_DIR = RUN_DIR / "f6_f7_full_image_demo"
shutil.rmtree(OUT_DIR, ignore_errors=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# OCR tuning
PSM = 6
MIN_CONF = 30.0

# Function 6 tuning (yours)
F6_PARAMS = dict(
    min_contour_area=200,
    canny_low=50,
    canny_high=150,
    hough_threshold=80,
    min_line_length=60,
    max_line_gap=10,
    min_bar_width=10,
    min_bar_height=10,
    min_bar_extent=0.60,
)

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def annotate_text_boxes(image_bgr, ocr_items):
    """Draw OCR boxes and token text onto a copy."""
    out = image_bgr.copy()
    for it in ocr_items:
        text = (it.get("text") or "").strip()
        left = it.get("left", None)
        top = it.get("top", None)
        w = it.get("width", None)
        h = it.get("height", None)
        if None in (left, top, w, h):
            continue

        x, y, ww, hh = int(left), int(top), int(w), int(h)
        cv2.rectangle(out, (x, y), (x + ww, y + hh), (0, 165, 255), 2)
        if text:
            cv2.putText(
                out,
                text[:20],
                (x, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
    return out

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Missing image: {IMAGE_PATH}")

    # Load original for text annotation overlay
    img_bgr = cv2.imread(str(IMAGE_PATH))
    if img_bgr is None:
        raise RuntimeError(f"Could not load image: {IMAGE_PATH}")

    # ------------------------------
    # Function 6 on FULL image
    # ------------------------------
    prims, prim_annot = detect_chart_primitives(
        image_path=str(IMAGE_PATH),
        **F6_PARAMS
    )

    prim_json = OUT_DIR / "full_primitives.json"
    prim_img = OUT_DIR / "full_annotated_primitives.png"
    prim_json.write_text(json.dumps(prims, indent=2), encoding="utf-8")
    cv2.imwrite(str(prim_img), prim_annot)

    # ------------------------------
    # Function 7 on FULL image
    # ------------------------------
    raw_text, ocr_items = extract_chart_text_ocr(
        image_path=str(IMAGE_PATH),
        psm=PSM,
        min_confidence=MIN_CONF
    )

    ocr_txt = OUT_DIR / "full_ocr_text.txt"
    ocr_json = OUT_DIR / "full_ocr_data.json"
    ocr_img = OUT_DIR / "full_annotated_text.png"

    ocr_txt.write_text(raw_text, encoding="utf-8")
    ocr_json.write_text(json.dumps(ocr_items, indent=2), encoding="utf-8")

    text_annot = annotate_text_boxes(img_bgr, ocr_items)
    cv2.imwrite(str(ocr_img), text_annot)

    # ------------------------------
    # Print summary
    # ------------------------------
    print("Saved outputs to:", OUT_DIR)
    print("Annotated primitives:", prim_img)
    print("Primitives JSON:", prim_json)
    print("Annotated text:", ocr_img)
    print("OCR text:", ocr_txt)
    print("OCR data:", ocr_json)
    print()
    print("Primitives detected:", len(prims))
    print("OCR tokens kept:", len(ocr_items))
    print("----- OCR RAW TEXT -----")
    print(raw_text.strip())

