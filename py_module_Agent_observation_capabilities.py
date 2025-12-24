# filename: py_module_Agent_observation_capabilities.py
"""
Module: py_module_Agent_observation_capabilities.py

Functions:
    1) get_png_attributes(png_path)
    2) crop_random_png(png_path, m, n)
    3) crop_at_location(png_path, x, y, m, n)
    4) detect_chart_elements(png_path)
    5) extract_objects(image_path, output_dir, color_tol=10, min_pixels=50, padding=4)
    6) detect_chart_primitives(
           image_path,
           min_contour_area=200,
           canny_low=50,
           canny_high=150,
           hough_threshold=80,
           min_line_length=60,
           max_line_gap=10,
           min_bar_width=10,
           min_bar_height=10,
           min_bar_extent=0.60
       )
    7) extract_chart_text_ocr(image_path, psm=6, min_confidence=0.0)
    8) clip_plot_type_evidence(
           image_path,
           plot_type_prompts=None,
           model_name="openai/clip-vit-base-patch32",
           device=None
       )
    9) crop_image_by_primitives_json(
           image_path,
           primitives_json_path,
           output_dir,
           include_types=("rectangle", "line"),
           padding=4,
           line_min_thickness=6,
           overwrite=True
       )
"""

from PIL import Image
import numpy as np
from pathlib import Path
import json
import os
import shutil

import cv2
import pytesseract
from scipy.ndimage import label, find_objects

import inspect

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
# FUNCTION 2: RANDOM CROP
# ============================================================

def crop_random_png(png_path, m, n):
    """
    Take a random crop of size m x n from a PNG image.

    The function does NOT save the crop to disk.

    Args:
        png_path (str or Path):
            Path to the PNG file.
        m (int):
            Crop width.
        n (int):
            Crop height.

    Returns:
        cropped_img (PIL.Image):
            Cropped image.
        metadata (dict):
            Crop location and size metadata.
    """
    attrs = get_png_attributes(png_path)
    img_width, img_height = attrs["width"], attrs["height"]

    if m > img_width or n > img_height:
        raise ValueError(
            f"Crop size ({m}x{n}) exceeds image size "
            f"({img_width}x{img_height})"
        )

    img = Image.open(png_path)

    x0 = np.random.randint(0, img_width - m + 1)
    y0 = np.random.randint(0, img_height - n + 1)

    crop_box = (x0, y0, x0 + m, y0 + n)
    cropped_img = img.crop(crop_box)

    metadata = {
        "input_png": str(png_path),
        "crop_width": m,
        "crop_height": n,
        "original_width": img_width,
        "original_height": img_height,
        "crop_top_left": {"x": x0, "y": y0},
        "crop_box": crop_box
    }

    return cropped_img, metadata


# ============================================================
# FUNCTION 3: CROP AT SPECIFIC LOCATION
# ============================================================

def crop_at_location(png_path, x, y, m, n):
    """
    Take a crop of size m x n from a PNG image at a fixed location.

    The function does NOT save the crop to disk.

    Args:
        png_path (str or Path):
            Path to the PNG file.
        x (int):
            Top-left x-coordinate.
        y (int):
            Top-left y-coordinate.
        m (int):
            Crop width.
        n (int):
            Crop height.

    Returns:
        cropped_img (PIL.Image):
            Cropped image.
        metadata (dict):
            Crop location and size metadata.
    """
    attrs = get_png_attributes(png_path)
    img_width, img_height = attrs["width"], attrs["height"]

    if x < 0 or y < 0:
        raise ValueError(
            f"Top-left coordinates must be non-negative. Got x={x}, y={y}"
        )

    if x + m > img_width or y + n > img_height:
        raise ValueError(
            f"Crop size ({m}x{n}) at location ({x},{y}) exceeds image bounds "
            f"({img_width}x{img_height})"
        )

    img = Image.open(png_path)
    crop_box = (x, y, x + m, y + n)
    cropped_img = img.crop(crop_box)

    metadata = {
        "input_png": str(png_path),
        "crop_width": m,
        "crop_height": n,
        "original_width": img_width,
        "original_height": img_height,
        "crop_top_left": {"x": x, "y": y},
        "crop_box": crop_box
    }

    return cropped_img, metadata


# ============================================================
# FUNCTION 4: DETECT CHART ELEMENTS (COARSE)
# ============================================================

def detect_chart_elements(png_path):
    """
    Detect coarse chart elements using classical CV + OCR.

    Elements detected:
        - Lines (axes / gridlines)
        - Rectangular blobs (bars or points)
        - Text labels (OCR-based)

    Args:
        png_path (str or Path):
            Path to the chart image.

    Returns:
        dict:
            {
                "lines": list,
                "points_or_bars": list,
                "labels": list
            }
    """
    img = cv2.imread(str(png_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Detect lines ---
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=50,
        maxLineGap=10
    )

    line_positions = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_positions.append({
                "start": (x1, y1),
                "end": (x2, y2)
            })

    # --- Detect bars / points ---
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    points_positions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 2 and h > 2:
            points_positions.append({
                "x": x,
                "y": y,
                "width": w,
                "height": h
            })

    # --- Detect text ---
    d = pytesseract.image_to_data(
        gray, output_type=pytesseract.Output.DICT
    )

    text_positions = []
    for i, text in enumerate(d["text"]):
        if text.strip():
            text_positions.append({
                "text": text,
                "x": d["left"][i],
                "y": d["top"][i],
                "width": d["width"][i],
                "height": d["height"][i]
            })

    return {
        "lines": line_positions,
        "points_or_bars": points_positions,
        "labels": text_positions
    }


# ============================================================
# FUNCTION 5: EXTRACT OBJECTS BY COLOR CONNECTIVITY
# ============================================================

def extract_objects(
    image_path,
    output_dir,
    color_tol=10,
    min_pixels=50,
    padding=4
):
    """
    Extract contiguous image regions based on color similarity.

    Each detected object is saved as a separate PNG.

    Args:
        image_path (str):
            Path to input image.
        output_dir (str):
            Directory where objects are saved.
        color_tol (int):
            Per-channel color tolerance.
        min_pixels (int):
            Minimum pixel count for an object.
        padding (int):
            Padding added around each object crop.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)

    pixels = arr.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)

    object_count = 0

    for color in unique_colors:
        mask = np.all(np.abs(arr - color) <= color_tol, axis=-1)

        if np.sum(mask) < min_pixels:
            continue

        labeled, _ = label(mask)
        slices = find_objects(labeled)

        for lbl, sl in enumerate(slices, start=1):
            if sl is None:
                continue

            coords = np.argwhere(labeled[sl] == lbl)
            if coords.shape[0] < min_pixels:
                continue

            min_y = max(sl[0].start - padding, 0)
            max_y = min(sl[0].stop + padding, arr.shape[0])
            min_x = max(sl[1].start - padding, 0)
            max_x = min(sl[1].stop + padding, arr.shape[1])

            cropped = img.crop((min_x, min_y, max_x, max_y))
            object_count += 1
            cropped.save(
                os.path.join(output_dir, f"object_{object_count}.png")
            )

    print(f"Extracted {object_count} objects to {output_dir}")

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
    import cv2
    import numpy as np

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
# FUNCTION 8: CLIP PLOT-TYPE EVIDENCE
# ============================================================

def clip_plot_type_evidence(
    image_path,
    plot_type_prompts=None,
    model_name="openai/clip-vit-base-patch32",
    device=None
):
    """
    Compute CLIP similarity scores between an image and a list of plot-type
    text hypotheses (prompts). Higher similarity = more evidence for that type.

    Args:
        image_path (str or Path):
            Path to the image file.
        plot_type_prompts (list[str] or None):
            Text hypotheses to compare against. If None, uses a default set.
        model_name (str):
            HuggingFace CLIP model name.
        device (str or None):
            "cuda", "cpu", or None (auto-detect).

    Returns:
        results (list[dict]):
            Sorted list (best first), each item:
            {
                "prompt": str,
                "score": float
            }
    """
    from PIL import Image
    import torch
    from transformers import CLIPProcessor, CLIPModel

    if plot_type_prompts is None:
        plot_type_prompts = [
            "a histogram",
            "a bar chart",
            "a line plot",
            "a scatter plot"
        ]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model + processor
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Encode image
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Encode text
    text_inputs = processor(
        text=plot_type_prompts,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Similarity (cosine, since normalized)
    similarity = (image_features @ text_features.T).squeeze(0)  # shape: (num_prompts,)

    results = [
        {"prompt": prompt, "score": float(score)}
        for prompt, score in zip(plot_type_prompts, similarity)
    ]
    results.sort(key=lambda d: d["score"], reverse=True)

    return results

# ============================================================
# FUNCTION 9 crop_image_by_primitives_json
# ============================================================

def crop_image_by_primitives_json(
    image_path,
    primitives_json_path,
    output_dir,
    include_types=("rectangle", "line"),
    padding=4,
    line_min_thickness=6,
    overwrite=True
):
    """
    Use primitives.json (from detect_chart_primitives) to crop regions out of an image.

    What it does:
        - Loads image_path (e.g., cropped.png)
        - Loads primitives_json_path (list of dict primitives)
        - For each primitive:
            * rectangle: uses bbox [x,y,w,h]
            * line: makes a bbox around start/end with a minimum thickness
        - Saves each crop into output_dir
        - Writes a crops_manifest.json with metadata for each saved crop

    Args:
        image_path (str or Path):
            Path to the source image to crop (e.g., cropped.png).
        primitives_json_path (str or Path):
            Path to primitives.json.
        output_dir (str or Path):
            Directory where crops are saved.
        include_types (tuple[str]):
            Which primitive types to crop ("rectangle", "line").
        padding (int):
            Extra pixels added around each crop (clamped to image bounds).
        line_min_thickness (int):
            Minimum thickness (in pixels) for line crops (helps avoid 1-px slices).
        overwrite (bool):
            If True, deletes and recreates output_dir for clean runs.

    Returns:
        manifest (list[dict]):
            One entry per saved crop (filename, primitive, crop_box, etc.).
    """
    import json
    import os
    import shutil
    import cv2

    image_path = str(image_path)
    primitives_json_path = str(primitives_json_path)
    output_dir = str(output_dir)

    if overwrite:
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Could not load image: {image_path}")

    H, W = img.shape[:2]

    with open(primitives_json_path, "r", encoding="utf-8") as f:
        primitives = json.load(f)

    def clamp(val, lo, hi):
        return max(lo, min(hi, val))

    def clamp_box(x1, y1, x2, y2):
        x1 = clamp(x1, 0, W)
        x2 = clamp(x2, 0, W)
        y1 = clamp(y1, 0, H)
        y2 = clamp(y2, 0, H)
        if x2 <= x1:
            x2 = min(W, x1 + 1)
        if y2 <= y1:
            y2 = min(H, y1 + 1)
        return x1, y1, x2, y2

    manifest = []
    rect_i = 0
    line_i = 0

    for p in primitives:
        ptype = p.get("type")
        if ptype not in include_types:
            continue

        # --------------------------
        # RECTANGLE CROPS
        # --------------------------
        if ptype == "rectangle":
            bbox = p.get("bbox", None)
            if not bbox or len(bbox) != 4:
                continue

            x, y, w, h = bbox
            x1 = int(x) - padding
            y1 = int(y) - padding
            x2 = int(x + w) + padding
            y2 = int(y + h) + padding
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2)

            crop = img[y1:y2, x1:x2].copy()

            rect_i += 1
            fname = f"rectangle_{rect_i:03d}.png"
            fpath = os.path.join(output_dir, fname)
            cv2.imwrite(fpath, crop)

            manifest.append({
                "filename": fname,
                "type": "rectangle",
                "source_image": os.path.basename(image_path),
                "primitive": p,
                "crop_box_xyxy": [x1, y1, x2, y2],
                "padding": int(padding)
            })

        # --------------------------
        # LINE CROPS
        # --------------------------
        elif ptype == "line":
            start = p.get("start", None)
            end = p.get("end", None)
            if not start or not end or len(start) != 2 or len(end) != 2:
                continue

            x1, y1 = map(int, start)
            x2, y2 = map(int, end)

            xmin = min(x1, x2)
            xmax = max(x1, x2)
            ymin = min(y1, y2)
            ymax = max(y1, y2)

            # Ensure line crops have thickness (avoid 1-px wide/tall crops)
            width = xmax - xmin
            height = ymax - ymin

            if width < line_min_thickness:
                extra = (line_min_thickness - width) // 2 + 1
                xmin -= extra
                xmax += extra

            if height < line_min_thickness:
                extra = (line_min_thickness - height) // 2 + 1
                ymin -= extra
                ymax += extra

            xmin -= padding
            ymin -= padding
            xmax += padding
            ymax += padding

            xmin, ymin, xmax, ymax = clamp_box(xmin, ymin, xmax, ymax)

            crop = img[ymin:ymax, xmin:xmax].copy()

            line_i += 1
            fname = f"line_{line_i:03d}.png"
            fpath = os.path.join(output_dir, fname)
            cv2.imwrite(fpath, crop)

            manifest.append({
                "filename": fname,
                "type": "line",
                "source_image": os.path.basename(image_path),
                "primitive": p,
                "crop_box_xyxy": [xmin, ymin, xmax, ymax],
                "padding": int(padding),
                "line_min_thickness": int(line_min_thickness)
            })

    # Save manifest
    manifest_path = os.path.join(output_dir, "crops_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest

# ============================================================
# FUNCTION 10: PRIMITIVES + TEXT (ONE JSON OUTPUT)
# ============================================================

def extract_primitives_and_text_to_json(
    image_path,
    output_json_path,
    primitives_kwargs=None,
    ocr_kwargs=None,
    add_line_bboxes=True,
    atomic_write=True
):
    """
    Run:
        - Function 6: detect_chart_primitives()
        - Function 7: extract_chart_text_ocr()

    And save ONE JSON file that contains:
        {
          "image": {...},
          "primitives": [...],
          "text": {
              "raw_text": "...",
              "items": [...]
          }
        }

    Args:
        image_path (str or Path):
            Path to the input image.
        output_json_path (str or Path):
            Where to write the combined JSON file.
        primitives_kwargs (dict or None):
            Extra kwargs passed into detect_chart_primitives().
            Example: {"min_bar_extent": 0.55, "hough_threshold": 70}
        ocr_kwargs (dict or None):
            Extra kwargs passed into extract_chart_text_ocr().
            Example: {"psm": 6, "min_confidence": 30.0}
        add_line_bboxes (bool):
            If True, adds bbox fields to line primitives for convenience.
        atomic_write (bool):
            If True, writes to a temporary file then os.replace().

    Returns:
        result (dict):
            The combined JSON-ready dictionary.
    """
    image_path = str(image_path)
    output_json_path = Path(output_json_path)

    if primitives_kwargs is None:
        primitives_kwargs = {}
    if ocr_kwargs is None:
        ocr_kwargs = {}

    # --- image size ---
    attrs = get_png_attributes(image_path)
    W, H = int(attrs["width"]), int(attrs["height"])

    filtered_kwargs = _filter_kwargs_for(
        detect_chart_primitives, primitives_kwargs
    )

    primitives, _annotated = detect_chart_primitives(
        image_path=image_path,
        **filtered_kwargs
    )

    # Optionally add bbox to line primitives
    if add_line_bboxes:
        for p in primitives:
            if p.get("type") == "line" and "start" in p and "end" in p:
                x1, y1 = p["start"]
                x2, y2 = p["end"]
                xmin, xmax = sorted([int(x1), int(x2)])
                ymin, ymax = sorted([int(y1), int(y2)])
                p["bbox_xyxy"] = [xmin, ymin, xmax, ymax]
                p["bbox_xywh"] = [xmin, ymin, max(1, xmax - xmin), max(1, ymax - ymin)]

    # --- run OCR (Function 7) ---
    raw_text, ocr_items = extract_chart_text_ocr(
        image_path=image_path,
        **ocr_kwargs
    )

    # Normalize OCR items to include bbox formats (in addition to left/top/width/height)
    ocr_items_norm = []
    for it in ocr_items:
        left = int(it.get("left", 0))
        top = int(it.get("top", 0))
        width = int(it.get("width", 0))
        height = int(it.get("height", 0))
        ocr_items_norm.append({
            **it,
            "bbox_xywh": [left, top, width, height],
            "bbox_xyxy": [left, top, left + width, top + height]
        })

    result = {
        "image": {
            "path": image_path,
            "width": W,
            "height": H,
            "color_mode": attrs.get("color_mode", None)
        },
        "primitives": primitives,
        "text": {
            "raw_text": raw_text,
            "items": ocr_items_norm
        },
        "params": {
            "primitives_kwargs": primitives_kwargs,
            "ocr_kwargs": ocr_kwargs,
            "add_line_bboxes": bool(add_line_bboxes)
        }
    }

    # --- write JSON ---
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    if atomic_write:
        tmp_path = output_json_path.with_suffix(output_json_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        os.replace(tmp_path, output_json_path)
    else:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result

# ============================================================
# FUNCTION 11 and helper functions: detect_primitives_text_aware_global_coords
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
    import cv2
    import json
    import shutil
    import numpy as np
    from pathlib import Path

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



import cv2

def annotate_axes_and_bars(
    image_path,
    inference,
    output_path
):
    import cv2

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


import json

def save_inference_json(inference, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(inference, f, indent=2)


# ============================================================
# FUNCTION 12: FULL BAR-CHART PIPELINE (TEXT-AWARE → INFERENCE)
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
    import json
    import cv2
    from pathlib import Path

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

    return {
        "image_path": str(image_path),
        "output_dir": str(output_dir),
        "detection": detection_result,
        "inference_json": str(inference_json_path),
        "annotated_image": str(annotated_path),
        "num_bars": len(inference.get("bars", [])),
        "num_axes": len(inference.get("axes", []))
    }
