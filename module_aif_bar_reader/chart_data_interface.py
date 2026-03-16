# filename: py_module_Agent_observation_capabilities.py

"""
Module: py_module_Agent_observation_capabilities.py

Purpose:
    Low-level image observation and chart-understanding utilities intended
    

Scope:
    - 

Public Functions:
    
"""


# ===============================
# Standard library IMPORTS
# ===============================
import os
import json
import math
import shutil
import inspect
from pathlib import Path
import re

# ===============================
# Third-party libraries
# ===============================
import numpy as np
import cv2
import pytesseract
from itertools import combinations

from PIL import Image
from scipy.ndimage import label, find_objects

# ============================================================
# INTERNAL HELPER: FILTER KWARGS FOR FUNCTION SIGNATURE
# ============================================================
def _filter_kwargs_for(func, kwargs):
    valid = set(inspect.signature(func).parameters.keys())
    return {k: v for k, v in kwargs.items() if k in valid}


# ============================================================
# FUNCTION 1: GET PNG ATTRIBUTES
# ============================================================

def get_png_attributes(png_path):
    """
    Read basic metadata from a PNG image.

    Args:
        png_path (str or Path):
            Path to the PNG file.

    Returns:
        dict:
            {
                "width": int,
                "height": int,
                "color_mode": str
            }
    """
    png_path = Path(png_path)

    with Image.open(png_path) as img:
        width, height = img.size
        color_mode = img.mode

    return {
        "width": width,
        "height": height,
        "color_mode": color_mode
    }



# ============================================================
# FUNCTION 6: detect_chart_primitives
# ============================================================

def detect_chart_primitives(
    image_path,
    canny_low=50,
    canny_high=150,
    hough_threshold=80,
    min_line_length=60,
    max_line_gap=10,
    angle_tol_deg=5.0
):
    """
    Detect low-level geometric primitives from a chart image.

    This version detects LINES ONLY.
    No rectangles, no bars, no semantic interpretation.

    Returns:
        primitives (list of dict)
        annotated_img (np.ndarray)
    """

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Could not load image: {image_path}")

    annotated_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    primitives = []

    # --------------------------------------------------
    # EDGE DETECTION
    # --------------------------------------------------
    edges = cv2.Canny(gray, canny_low, canny_high)

    # --------------------------------------------------
    # LINE DETECTION (Probabilistic Hough)
    # --------------------------------------------------
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    if lines is None:
        return [], annotated_img

    # --------------------------------------------------
    # LINE FILTERING + CLASSIFICATION
    # --------------------------------------------------
    for line in lines:
        x1, y1, x2, y2 = map(int, line[0])

        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))

        if length < min_line_length:
            continue

        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)

        # Normalize angle to [-90, 90]
        if angle_deg < -90:
            angle_deg += 180
        elif angle_deg > 90:
            angle_deg -= 180

        orientation = None
        if abs(angle_deg) <= angle_tol_deg:
            orientation = "horizontal"
        elif abs(abs(angle_deg) - 90) <= angle_tol_deg:
            orientation = "vertical"
        else:
            continue  # discard diagonal noise

        primitive = {
            "type": "line",
            "orientation": orientation,
            "start": [x1, y1],
            "end": [x2, y2],
            "length": length,
            "angle_deg": angle_deg
        }

        primitives.append(primitive)

        # Visualization
        color = (255, 0, 0) if orientation == "horizontal" else (0, 255, 0)
        cv2.line(annotated_img, (x1, y1), (x2, y2), color, 2)

    return primitives, annotated_img


# ============================================================
# FUNCTION 7: EXTRACT CHART TEXT (OCR)
# ============================================================

def extract_chart_text_ocr(
    image_path,
    psm=6,
    min_confidence=0.0
):
    """
    Extract textual content from a chart image using Tesseract OCR.

    This function is intentionally layout-agnostic.
    Layout reasoning (axes, ticks, rotation) must be done downstream.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    config = f"--psm {psm}"

    raw_text = pytesseract.image_to_string(gray, config=config)
    data = pytesseract.image_to_data(
        gray, output_type=pytesseract.Output.DICT, config=config
    )

    ocr_items = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = float(data["conf"][i])
        if text and conf >= min_confidence:
            ocr_items.append({
                "text": text,
                "left": int(data["left"][i]),
                "top": int(data["top"][i]),
                "width": int(data["width"][i]),
                "height": int(data["height"][i]),
                "confidence": conf
            })

    return raw_text, ocr_items


# ============================================================
# INTERNAL HELPER: OFFSET PRIMITIVES TO GLOBAL COORDINATES
# ============================================================

def _offset_primitives(primitives, dx, dy):
    """
    Shift primitives detected in a cropped ROI back to full-image coordinates.

    Args:
        primitives (list[dict]): output of detect_chart_primitives on ROI
        dx (int): x-offset of ROI (left)
        dy (int): y-offset of ROI (top)

    Returns:
        list[dict]: primitives with coordinates in original image frame
    """
    out = []

    for p in primitives:
        p = dict(p)  # shallow copy

        if p["type"] == "line":
            x1, y1 = p["start"]
            x2, y2 = p["end"]
            p["start"] = [x1 + dx, y1 + dy]
            p["end"]   = [x2 + dx, y2 + dy]

        out.append(p)

    return out

# ============================================================
# FUNCTION 11: TEXT-AWARE PRIMITIVE DETECTION (GLOBAL COORDS)
# ============================================================

def detect_primitives_text_aware_global_coords(
    image_path,
    output_dir,
    primitives_kwargs=None,
    ocr_kwargs=None,
    group_tol=5
):
    """
    Detect chart primitives while excluding text regions,
    but return ALL coordinates in the original image frame.

    Outputs (same contract as your demo):
        - annotated_primitives.png
        - primitives.json
        - annotated_text.png
        - ocr_data.json
    """

    if primitives_kwargs is None:
        primitives_kwargs = {}
    if ocr_kwargs is None:
        ocr_kwargs = {}

    image_path = Path(image_path)
    output_dir = Path(output_dir)

    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Load image
    # --------------------------------------------------
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Could not load image: {image_path}")
    H, W = img.shape[:2]

    # --------------------------------------------------
    # OCR FIRST (full image)
    # --------------------------------------------------
    raw_text, ocr_items = extract_chart_text_ocr(
        image_path=str(image_path),
        **ocr_kwargs
    )

    # Normalize OCR boxes
    boxes = []
    for it in ocr_items:
        x1 = int(it["left"])
        y1 = int(it["top"])
        x2 = x1 + int(it["width"])
        y2 = y1 + int(it["height"])
        boxes.append((x1, y1, x2, y2))


    # --------------------------------------------------
    # GROUP OCR BOXES INTO TEXT BANDS
    # --------------------------------------------------
    def group_by_pair(boxes, key_fn, tol):
        groups = []
        keys = []

        for b in boxes:
            k = key_fn(b)
            matched = False
            for i, gk in enumerate(keys):
                if abs(gk[0] - k[0]) <= tol and abs(gk[1] - k[1]) <= tol:
                    groups[i].append(b)
                    matched = True
                    break
            if not matched:
                keys.append(k)
                groups.append([b])

        return groups


# --------------------------------------------------
# VERTICAL TEXT: y-axis label vs y-tick labels
# (same left/right bounds)
# --------------------------------------------------
    vertical_groups = group_by_pair(
        boxes,
        key_fn=lambda b: (b[0], b[2]),   # (left, right)
        tol=group_tol
    )

# --------------------------------------------------
# VERTICAL TEXT GROUPS
# --------------------------------------------------
    def vertical_span(group):
        ys = [b[1] for b in group] + [b[3] for b in group]
        return max(ys) - min(ys)

    # y-tick labels → MANY boxes, BIG vertical span
    y_tick_group = max(vertical_groups, key=vertical_span)
    y_tick_right = max(b[2] for b in y_tick_group)

    # y-axis label → remaining group, leftmost
    y_axis_label_group = min(
        (g for g in vertical_groups if g is not y_tick_group),
        key=lambda g: min(b[0] for b in g)
    )

    pad = 8  # pixels, tune 6–12 if needed

    x1 = max(0, min(b[0] for b in y_axis_label_group) - pad)
    y1 = max(0, min(b[1] for b in y_axis_label_group) - pad)
    x2 = min(W, max(b[2] for b in y_axis_label_group) + pad)
    y2 = min(H, max(b[3] for b in y_axis_label_group) + pad)


    crop = img[y1:y2, x1:x2].copy()
    rot = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

    fixed_text = pytesseract.image_to_string(
        rot, config="--psm 6"
    ).strip()

    ocr_items = [
        it for it in ocr_items
        if not (x1 <= it["left"] <= x2 and y1 <= it["top"] <= y2)
    ]

    ocr_items.append({
        "text": fixed_text,
        "left": x1,
        "top": y1,
        "width": x2 - x1,
        "height": y2 - y1,
        "confidence": 100.0,
        "orientation": "vertical"
    })



# --------------------------------------------------
# HORIZONTAL TEXT: title, x-axis label, x-tick labels
# (same top/bottom bounds)
# --------------------------------------------------
    horizontal_groups = group_by_pair(
        boxes,
        key_fn=lambda b: (b[1], b[3]),   # (top, bottom)
        tol=group_tol
    )

# --------------------------------------------------
# Identify title (topmost band)
# --------------------------------------------------
    title_group = min(
        horizontal_groups,
        key=lambda g: min(b[1] for b in g)
    )
    title_lower = max(b[3] for b in title_group)

# --------------------------------------------------
# Identify x-tick labels:
# bottom-half + largest horizontal span
# --------------------------------------------------
    def horizontal_span(group):
        xs = [b[0] for b in group] + [b[2] for b in group]
        return max(xs) - min(xs)

    # Consider only groups below the vertical midpoint
    bottom_candidates = [
        g for g in horizontal_groups
        if min(b[1] for b in g) > H * 0.5
    ]

    x_tick_group = max(bottom_candidates, key=horizontal_span)
    x_tick_upper = min(b[1] for b in x_tick_group)



    # (optional debug – remove after validation)
    print("Vertical group sizes:", [len(g) for g in vertical_groups])
    print("Horizontal group sizes:", [len(g) for g in horizontal_groups])
    print("y_tick_right:", y_tick_right)
    print("title_lower:", title_lower)
    print("x_tick_upper:", x_tick_upper)


    # --------------------------------------------------
    # Define plot ROI (NO whitening)
    # --------------------------------------------------
    roi_x1 = y_tick_right
    roi_y1 = title_lower
    roi_x2 = W
    roi_y2 = x_tick_upper


    plot_roi = img[roi_y1:roi_y2, roi_x1:roi_x2].copy()

    # --------------------------------------------------
    # Run OpenCV on ROI
    # --------------------------------------------------
    tmp_roi = output_dir / "_plot_roi.png"
    cv2.imwrite(str(tmp_roi), plot_roi)

    filtered_kwargs = _filter_kwargs_for(
        detect_chart_primitives, primitives_kwargs
    )

    primitives_roi, annotated_roi = detect_chart_primitives(
        image_path=str(tmp_roi),
        **filtered_kwargs
    )


    # Translate primitives back to global coords
    primitives = _offset_primitives(
        primitives_roi,
        dx=roi_x1,
        dy=roi_y1
    )

    # --------------------------------------------------
    # Annotate primitives on full image
    # --------------------------------------------------
    annotated_full = img.copy()

    for p in primitives:
        if p["type"] == "line":
            x1, y1 = p["start"]
            x2, y2 = p["end"]
            cv2.line(
                annotated_full,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------
    with open(output_dir / "primitives.json", "w", encoding="utf-8") as f:
        json.dump(primitives, f, indent=2)

    cv2.imwrite(
        str(output_dir / "annotated_primitives.png"),
        annotated_full
    )

    # OCR visualization
    ocr_vis = img.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(ocr_vis, (x1, y1), (x2, y2), (0, 165, 255), 2)

    with open(output_dir / "ocr_data.json", "w", encoding="utf-8") as f:
        json.dump(ocr_items, f, indent=2)

    cv2.imwrite(
        str(output_dir / "annotated_text.png"),
        ocr_vis
    )

    return {
        "y_tick_right": int(y_tick_right),
        "x_tick_upper": int(x_tick_upper),
        "title_lower": int(title_lower),
        "plot_roi_xyxy": [roi_x1, roi_y1, roi_x2, roi_y2],
        "num_primitives": len(primitives)
    }

# ============================================================
# FUNCTION 12: INFER AXES AND BARS FROM PRIMITIVES
# ============================================================

def infer_axes_and_bars_from_primitives(
    primitives,
    image_width,
    image_height,
    y_tol=6
):
    horizontals = [p for p in primitives if p["orientation"] == "horizontal"]
    verticals   = [p for p in primitives if p["orientation"] == "vertical"]

    # --- axes ---
    x_axis = max(horizontals, key=lambda p: (p["start"][1], p["length"]))
    y_axis = min(verticals, key=lambda p: (p["start"][0], -p["length"]))
    baseline_y = x_axis["start"][1]

    # --- bar tops ---
    bar_tops = [
        h for h in horizontals
        if h is not x_axis
        and h["length"] < 0.4 * x_axis["length"]
        and h["start"][1] < baseline_y - y_tol
    ]

    bar_tops.sort(key=lambda h: min(h["start"][0], h["end"][0]))

    bars = [
        {
            "top_y": int(h["start"][1]),
            "height_px": int(baseline_y - h["start"][1]),
            "x_range": sorted([h["start"][0], h["end"][0]])
        }
        for h in bar_tops
    ]

    return {
        "axes": {
            "x_axis": x_axis,
            "y_axis": y_axis
        },
        "bars": bars
    }

# ============================================================
# FUNCTION 13: ANNOTATE AXES AND BARS ON IMAGE
# ============================================================

def annotate_axes_and_bars(
    image_path,
    inference,
    output_path
):

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Could not load image: {image_path}")

    # --- draw axes ---
    x_axis = inference["axes"]["x_axis"]
    y_axis = inference["axes"]["y_axis"]

    cv2.line(
        img,
        tuple(x_axis["start"]),
        tuple(x_axis["end"]),
        (0, 0, 255),
        3
    )

    cv2.line(
        img,
        tuple(y_axis["start"]),
        tuple(y_axis["end"]),
        (0, 0, 255),
        3
    )

    baseline_y = x_axis["start"][1]

    # --- draw bars (reconstructed bbox) ---
    for bar in inference["bars"]:
        x1, x2 = bar["x_range"]
        y1 = bar["top_y"]
        y2 = baseline_y

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    cv2.imwrite(str(output_path), img)

# ============================================================
# FUNCTION 14: SAVE INFERENCE JSON
# ============================================================

def save_inference_json(inference, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(inference, f, indent=2)

# ============================================================
# FUNCTION 15: ANNOTATE TEXT + AXES + BARS
# ============================================================

def annotate_text_and_axes_and_bars(
    image_path,
    ocr_items,
    inference,
    output_path
):
    """
    Draw OCR boxes + inferred axes + inferred bars on the same image.

    Args:
        image_path (str or Path)
        ocr_items (list[dict]): from ocr_data.json
        inference (dict): from inferred_axes_and_bars.json
        output_path (str or Path)
    """

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Could not load image: {image_path}")

    # --------------------------------------------------
    # Draw OCR boxes (orange)
    # --------------------------------------------------
    for it in ocr_items:
        x1 = int(it["left"])
        y1 = int(it["top"])
        x2 = x1 + int(it["width"])
        y2 = y1 + int(it["height"])

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 165, 255),  # orange
            2
        )

    # --------------------------------------------------
    # Draw axes (blue)
    # --------------------------------------------------
    axes = inference.get("axes", {})
    for ax in axes.values():
        x1, y1 = ax["start"]
        x2, y2 = ax["end"]

        cv2.line(
            img,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (255, 0, 0),  # blue
            3
        )

    # --------------------------------------------------
    # Draw bars (green)
    # --------------------------------------------------
    for bar in inference.get("bars", []):
        x1, x2 = bar["x_range"]
        top_y = bar["top_y"]
        height = bar["height_px"]

        y_bottom = top_y + height

        cv2.rectangle(
            img,
            (int(x1), int(top_y)),
            (int(x2), int(y_bottom)),
            (0, 255, 0),  # green
            2
        )

    cv2.imwrite(str(output_path), img)
 

def image_interpretation_output_to_agent(
    axes_and_bars_json_path: str,
    ocr_json_path: str
):
    """
    Parameters
    ----------
    axes_and_bars_json_path : str
        Path to inferred_axes_and_bars.json
    ocr_json_path : str
        Path to ocr_data.json

    Returns
    -------
    bar_heights : list[float]
        Bar heights in data units (e.g. USD)
    tick_values : list[float]
        Y-axis tick values
    number_of_ticks : int
    number_of_bars : int
    """

    # -----------------------------
    # Load JSON files
    # -----------------------------
    with open(axes_and_bars_json_path, "r") as f:
        axes_and_bars = json.load(f)

    with open(ocr_json_path, "r") as f:
        ocr_data = json.load(f)

    bars = axes_and_bars["bars"]

    # -----------------------------
    # Extract numeric Y-axis ticks
    # -----------------------------
    tick_entries = []
    for item in ocr_data:
        text = item["text"].strip()
        if re.fullmatch(r"\d+(\.\d+)?", text):
            value = float(text)
            y_center = item["top"] + item["height"] / 2
            tick_entries.append((y_center, value))

    if len(tick_entries) < 2:
        raise ValueError("Not enough Y-axis ticks to infer scale.")

    # Sort top → bottom in image space
    tick_entries.sort(key=lambda x: x[0])

    tick_pixels = np.array([t[0] for t in tick_entries])
    tick_values = np.array([t[1] for t in tick_entries])

    number_of_ticks = len(tick_values)

    # -----------------------------
    # Fit pixel → value mapping
    # -----------------------------
    a, b = np.polyfit(tick_pixels, tick_values, 1)

    def pixel_to_value(y_pixel):
        return a * y_pixel + b

    # -----------------------------
    # Baseline (y = 0)
    # -----------------------------
    baseline_pixel = tick_pixels.max()
    baseline_value = pixel_to_value(baseline_pixel)

    # -----------------------------
    # Compute bar heights (FIXED)
    # -----------------------------
    bar_heights = []
    for bar in bars:
        top_y = bar["top_y"]

        value_top = pixel_to_value(top_y)
        bar_value = value_top - baseline_value

        bar_heights.append(round(bar_value, 2))

    number_of_bars = len(bar_heights)

    return bar_heights, tick_values.tolist(), number_of_ticks, number_of_bars

def _round_floats(obj, decimals=4):
    """Recursively round all floats in a nested structure."""
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, dict):
        return {k: _round_floats(v, decimals) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, decimals) for v in obj]
    return obj

def summarize_bar_chart_to_json(
    bar_dict: dict,
    output_dir: str,
    filename: str = "bar_chart_summary.json"
):
    """
    Compute all valid summary statistics from a bar chart dictionary
    and save the result as a JSON file.
    """

    if not bar_dict:
        raise ValueError("bar_dict is empty")

    if any(v < 0 for v in bar_dict.values()):
        raise ValueError("Bar values must be non-negative")

    categories = list(bar_dict.keys())
    values = list(bar_dict.values())

    n_categories = len(categories)
    total = sum(values)

    proportions = {
        k: (v / total if total > 0 else 0.0)
        for k, v in bar_dict.items()
    }

    max_cat = max(bar_dict, key=bar_dict.get)
    min_cat = min(bar_dict, key=bar_dict.get)

    mean = total / n_categories
    variance = sum((v - mean) ** 2 for v in values) / n_categories
    std = math.sqrt(variance)

    entropy = -sum(
        p * math.log(p) for p in proportions.values() if p > 0
    )

    sorted_vals = sorted(values)
    gini_num = sum(
        (2 * i - n_categories - 1) * v
        for i, v in enumerate(sorted_vals, start=1)
    )
    gini = gini_num / (n_categories * total) if total > 0 else 0.0

    hhi = sum(p ** 2 for p in proportions.values())

    sorted_items = sorted(bar_dict.items(), key=lambda x: x[1], reverse=True)

    cumulative = []
    running = 0
    for k, v in sorted_items:
        running += v
        cumulative.append({
            "category": k,
            "cumulative_share": running / total if total > 0 else 0.0
        })

    pairwise = []
    for (k1, v1), (k2, v2) in combinations(bar_dict.items(), 2):
        pairwise.append({
            "category_a": k1,
            "category_b": k2,
            "difference": v1 - v2,
            "ratio": (v1 / v2) if v2 != 0 else None
        })

    summary = {
        "meta": {
            "n_categories": n_categories,
            "total": total
        },
        "per_category": {
            k: {
                "value": bar_dict[k],
                "proportion": proportions[k],
                "rank": sorted_items.index((k, bar_dict[k])) + 1
            }
            for k in categories
        },
        "central_tendency": {
            "mean": mean,
            "mode_category": max_cat
        },
        "dispersion": {
            "range": bar_dict[max_cat] - bar_dict[min_cat],
            "variance": variance,
            "std": std,
            "coefficient_of_variation": std / mean if mean != 0 else None
        },
        "inequality": {
            "entropy": entropy,
            "gini": gini,
            "hhi": hhi
        },
        "concentration": {
            "top_category_share": proportions[max_cat],
            "top_2_share": (
                sum(v for _, v in sorted_items[:2]) / total
                if total > 0 else 0.0
            ),
            "cumulative_share": cumulative
        },
        "pairwise_comparisons": pairwise
    }

    # 🔹 Round floats to 4 decimal places before saving
    summary = _round_floats(summary, decimals=4)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ============================================================
# FUNCTION 16: FULL BAR-CHART PIPELINE (END-TO-END)
# ============================================================

def run_bar_chart_full_pipeline(
    image_path,
    output_dir,
    primitives_kwargs=None,
    ocr_kwargs=None
):
    """
    End-to-end bar chart pipeline:
      1) Text-aware primitive detection (global coords)
      2) Axis + bar inference
      3) JSON outputs
      4) Annotated images

    Args:
        image_path (str):
            Path to chart image.
        output_dir (str or Path):
            Output directory.
        primitives_kwargs (dict):
            Passed to detect_chart_primitives.
        ocr_kwargs (dict):
            Passed to extract_chart_text_ocr.

    Returns:
        dict:
            Summary with paths and inference results.
    """

    if primitives_kwargs is None:
        primitives_kwargs = {}
    if ocr_kwargs is None:
        ocr_kwargs = {}

    output_dir = Path(output_dir)

    # --------------------------------------------------
    # STEP 1: Text-aware primitive detection
    # --------------------------------------------------
    detection_result = detect_primitives_text_aware_global_coords(
        image_path=image_path,
        output_dir=output_dir,
        primitives_kwargs=primitives_kwargs,
        ocr_kwargs=ocr_kwargs
    )

    # --------------------------------------------------
    # STEP 2: Load primitives
    # --------------------------------------------------
    with open(output_dir / "primitives.json", "r", encoding="utf-8") as f:
        primitives = json.load(f)

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Could not load image: {image_path}")

    H, W = img.shape[:2]

    # --------------------------------------------------
    # STEP 3: Infer axes and bars
    # --------------------------------------------------
    inference = infer_axes_and_bars_from_primitives(
        primitives,
        image_width=W,
        image_height=H
    )

    # --------------------------------------------------
    # STEP 4: Save inference JSON
    # --------------------------------------------------
    inference_json_path = output_dir / "inferred_axes_and_bars.json"
    save_inference_json(inference, inference_json_path)

    # --------------------------------------------------
    # STEP 5: Annotate image
    # --------------------------------------------------
    annotated_path = output_dir / "annotated_axes_and_bars.png"
    annotate_axes_and_bars(
        image_path=image_path,
        inference=inference,
        output_path=annotated_path
    )

    # --------------------------------------------------
    # STEP 6: Combined annotation
    # --------------------------------------------------
    with open(output_dir / "ocr_data.json", "r", encoding="utf-8") as f:
        ocr_items = json.load(f)

    combined_path = output_dir / "annotated_combined.png"

    annotate_text_and_axes_and_bars(
        image_path=image_path,
        ocr_items=ocr_items,
        inference=inference,
        output_path=combined_path
    )

    return {
        "image_path": str(image_path),
        "output_dir": str(output_dir),
        "detection": detection_result,
        "inference_json": str(inference_json_path),
        "annotated_image": str(annotated_path),
        "num_bars": len(inference.get("bars", [])),
        "num_axes": len(inference.get("axes", []))
    }

def extract_chart_data_for_agent(png_path, output_dir):

    output_dir = Path(output_dir)

    run_bar_chart_full_pipeline(
        image_path=png_path,
        output_dir=output_dir
    )

    bar_values, tick_values, n_ticks, n_bars = \
        image_interpretation_output_to_agent(
            axes_and_bars_json_path=output_dir / "inferred_axes_and_bars.json",
            ocr_json_path=output_dir / "ocr_data.json"
        )

    return {
    "bar_values": bar_values,
    "tick_values": tick_values,
    "n_ticks": n_ticks,
    "n_bars": n_bars,
    "files": {
        "inference_json": output_dir / "inferred_axes_and_bars.json",
        "ocr_json": output_dir / "ocr_data.json",
        "primitives_json": output_dir / "primitives.json",
        "annotated_image": output_dir / "annotated_axes_and_bars.png"
        }
    }