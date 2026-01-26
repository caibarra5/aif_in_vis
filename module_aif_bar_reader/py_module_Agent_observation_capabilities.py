# filename: py_module_Agent_observation_capabilities.py

"""
Module: py_module_Agent_observation_capabilities.py

Purpose:
    Low-level image observation and chart-understanding utilities intended
    for agent perception pipelines. The module combines classical computer
    vision, OCR, and vision–language model scoring to extract structural
    evidence from chart-like images, with a focus on bar charts.

Scope:
    - Image metadata and cropping utilities
    - Primitive detection (lines, rectangles) via OpenCV
    - OCR-based text extraction and normalization
    - CLIP-based plot-type evidence scoring
    - JSON-based interoperability between pipeline stages
    - End-to-end bar-chart inference (axes, bars, values)

Public Functions:
    1) get_png_attributes(png_path)
        Read basic image metadata (width, height, color mode).

    2) crop_random_png(png_path, m, n)
        Randomly crop an image without saving to disk.

    3) crop_at_location(png_path, x, y, m, n)
        Crop an image at a fixed pixel location.

    4) detect_chart_elements(png_path)
        Coarse, heuristic detection of lines, blobs, and text (exploratory).

    5) extract_objects(image_path, output_dir, color_tol=10,
                       min_pixels=50, padding=4)
        Extract connected color regions as separate images.

    6) detect_chart_primitives(image_path, ...)
        Detect low-level geometric primitives (currently lines only)
        using edge detection and Hough transforms.

    7) extract_chart_text_ocr(image_path, psm=6, min_confidence=0.0)
        Extract textual elements using Tesseract OCR.

    8) clip_plot_type_evidence(image_path, plot_type_prompts=None,
                               model_name="openai/clip-vit-base-patch32",
                               device=None)
        Score image similarity against plot-type text prompts using CLIP.

    9) crop_image_by_primitives_json(image_path, primitives_json_path,
                                     output_dir, ...)
        Crop image regions based on detected primitives stored in JSON.

    10) extract_primitives_and_text_to_json(image_path, output_json_path, ...)
        Run primitive detection and OCR and emit a single combined JSON file.

    11) detect_primitives_text_aware_global_coords(image_path, output_dir, ...)
        Detect primitives while excluding text regions, returning global
        image coordinates.

    12) infer_axes_and_bars_from_primitives(primitives, image_width,
                                            image_height, ...)
        Infer chart axes and bar geometry from detected primitives.

    13) annotate_axes_and_bars(image_path, inference, output_path)
        Draw inferred axes and bars onto an image.

    14) save_inference_json(inference, output_path)
        Serialize inferred chart structure to JSON.

    15) annotate_text_and_axes_and_bars(image_path, ocr_items,
                                        inference, output_path)
        Visualize OCR boxes, axes, and bars together.

    16) run_bar_chart_full_pipeline(image_path, output_dir, ...)
        End-to-end bar-chart processing pipeline.

    17) map_bar_heights_to_xlabels_from_jsons(ocr_json_path,
                                              inferred_json_path)
        Map bar heights to x-axis labels using OCR and inferred geometry.

Notes:
    - No learning or model training occurs in this module.
    - Most functions are deterministic and stateless.
    - Higher-level semantic reasoning is expected to be handled downstream.
"""


# ===============================
# Standard library
# ===============================
import os
import json
import math
import shutil
import inspect
from pathlib import Path

# ===============================
# Third-party libraries
# ===============================
import numpy as np
import cv2
import pytesseract

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
# FUNCTION 4: DETECT CHART ELEMENTS (COARSE, NOT RELIABLE)
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
    import torch
    from transformers import CLIPProcessor, CLIPModel

    if plot_type_prompts is None:
        plot_type_prompts = [
            "a bar chart",
            "a line chart",
            "a scatter plot",
            "a histogram",
            "a pie chart",
            "a heatmap",
            "a box plot",
            "a violin plot",
            "an area chart",
            "a radar chart"
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
# FUNCTION 9: CROP IMAGE BY PRIMITIVES (JSON INPUT)
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


# ============================================================
# FUNCTION 17: MAP BAR HEIGHTS TO X-LABELS
# ============================================================

def map_bar_heights_to_xlabels_from_jsons(
    ocr_json_path,
    inferred_json_path
):
    """
    Map bar heights to x-axis labels using OCR and inferred primitives.

    Args:
        ocr_json_path (str): path to ocr_data.json
        inferred_json_path (str): path to inferred_axes_and_bars.json

    Returns:
        dict[str, float]: {x_label: bar_height}
    """

    # --------------------------------------------------
    # Load JSONs
    # --------------------------------------------------
    with open(ocr_json_path, "r", encoding="utf-8") as f:
        ocr_items = json.load(f)

    with open(inferred_json_path, "r", encoding="utf-8") as f:
        inferred = json.load(f)

    # --------------------------------------------------
    # Extract y-axis numeric ticks
    # --------------------------------------------------
    y_ticks = []
    for it in ocr_items:
        try:
            val = float(it["text"])
        except ValueError:
            continue

        y_mid = it["top"] + it["height"] / 2
        y_ticks.append((y_mid, val))

    if len(y_ticks) < 2:
        raise RuntimeError("Not enough y-axis ticks to infer scale")

    # sort top → bottom
    y_ticks.sort(key=lambda t: t[0])

    # --------------------------------------------------
    # Pixel → value linear mapping
    # --------------------------------------------------
    y_pixels = [t[0] for t in y_ticks]
    y_values = [t[1] for t in y_ticks]

    a = (y_values[-1] - y_values[0]) / (y_pixels[-1] - y_pixels[0])
    b = y_values[0] - a * y_pixels[0]

    def pixel_to_value(y_px):
        return a * y_px + b

    # --------------------------------------------------
    # Extract x-axis labels
    # --------------------------------------------------
    x_labels = []
    for it in ocr_items:
        if not it["text"].replace(".", "").isdigit():
            x_mid = it["left"] + it["width"] / 2
            x_labels.append((x_mid, it["text"]))

    # --------------------------------------------------
    # Match bars to x-labels
    # --------------------------------------------------
    result = {}

    for bar in inferred["bars"]:
        bar_mid_x = sum(bar["x_range"]) / 2

        label_mid, label = min(
            x_labels,
            key=lambda t: abs(t[0] - bar_mid_x)
        )

        bar_value = pixel_to_value(bar["top_y"])
        result[label] = int(round(bar_value, 2))

    return result

# ============================================================
# FUNCTION 18: INFER TEXT STRUCTURE FROM RAW OCR
# ============================================================

def infer_text_structure_from_ocr(
    ocr_json_path,
    output_json_path,
    group_tol_px=6
):
    """
    Infer chart text structure from raw OCR detections.

    Identifies:
      - title
      - x-axis label
      - y-axis label
      - x-axis tick labels
      - y-axis tick labels
      - spacing statistics

    Args:
        ocr_json_path (str or Path):
            Path to raw OCR JSON (list of items).
        output_json_path (str or Path):
            Path to save structured text JSON.
        group_tol_px (int):
            Pixel tolerance for grouping aligned text.

    Returns:
        dict: inferred structure
    """
    import statistics

    ocr_json_path = Path(ocr_json_path)
    output_json_path = Path(output_json_path)

    with open(ocr_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    # ---------------------------------------------
    # Helpers
    # ---------------------------------------------
    def bbox_xyxy(it):
        x1 = it["left"]
        y1 = it["top"]
        x2 = x1 + it["width"]
        y2 = y1 + it["height"]
        return x1, y1, x2, y2

    def center_x(it):
        x1, _, x2, _ = bbox_xyxy(it)
        return (x1 + x2) / 2

    def center_y(it):
        _, y1, _, y2 = bbox_xyxy(it)
        return (y1 + y2) / 2

    # ---------------------------------------------
    # Vertical extent reference
    # ---------------------------------------------
    all_y = [center_y(it) for it in items]
    min_y, max_y = min(all_y), max(all_y)
    height = max_y - min_y

    # ---------------------------------------------
    # Title: highest text band (topmost)
    # ---------------------------------------------
    title_items = [
        it for it in items
        if center_y(it) <= min_y + 0.15 * height
    ]

    # ---------------------------------------------
    # X-axis label: bottom-most centered text
    # ---------------------------------------------
    x_axis_label_items = [
        it for it in items
        if center_y(it) >= min_y + 0.9 * height
    ]

    # ---------------------------------------------
    # Y-axis label: vertical orientation OR leftmost tall text
    # ---------------------------------------------
    y_axis_label_items = [
        it for it in items
        if it.get("orientation") == "vertical"
    ]

    # ---------------------------------------------
    # Group by aligned X (for y-ticks)
    # ---------------------------------------------
    def group_by_x(items):
        groups = []
        for it in items:
            cx = center_x(it)
            placed = False
            for g in groups:
                if abs(center_x(g[0]) - cx) <= group_tol_px:
                    g.append(it)
                    placed = True
                    break
            if not placed:
                groups.append([it])
        return groups

    # ---------------------------------------------
    # Group by aligned Y (for x-ticks)
    # ---------------------------------------------
    def group_by_y(items):
        groups = []
        for it in items:
            cy = center_y(it)
            placed = False
            for g in groups:
                if abs(center_y(g[0]) - cy) <= group_tol_px:
                    g.append(it)
                    placed = True
                    break
            if not placed:
                groups.append([it])
        return groups

    # ---------------------------------------------
    # Y-axis tick labels (numbers on left side)
    # ---------------------------------------------
    left_items = [
        it for it in items
        if center_x(it) <= min(center_x(i) for i in items) + 0.25 * (
            max(center_x(i) for i in items) - min(center_x(i) for i in items)
        )
        and it not in y_axis_label_items
    ]

    y_tick_groups = group_by_x(left_items)
    y_tick_group = max(y_tick_groups, key=len) if y_tick_groups else []

    y_tick_centers = sorted(center_y(it) for it in y_tick_group)
    y_tick_spacings = [
        y_tick_centers[i + 1] - y_tick_centers[i]
        for i in range(len(y_tick_centers) - 1)
    ]

    # ---------------------------------------------
    # X-axis tick labels (spread horizontally near bottom)
    # ---------------------------------------------
    bottom_items = [
        it for it in items
        if center_y(it) >= min_y + 0.75 * height
        and it not in x_axis_label_items
    ]

    x_tick_groups = group_by_y(bottom_items)
    x_tick_group = max(x_tick_groups, key=len) if x_tick_groups else []

    x_tick_centers = sorted(center_x(it) for it in x_tick_group)
    x_tick_spacings = [
        x_tick_centers[i + 1] - x_tick_centers[i]
        for i in range(len(x_tick_centers) - 1)
    ]

    # ---------------------------------------------
    # Assemble output
    # ---------------------------------------------
    result = {
        "title": {
            "items": title_items,
            "text": " ".join(it["text"] for it in title_items)
        },
        "x_axis_label": {
            "items": x_axis_label_items,
            "text": " ".join(it["text"] for it in x_axis_label_items)
        },
        "y_axis_label": {
            "items": y_axis_label_items,
            "text": " ".join(it["text"] for it in y_axis_label_items)
        },
        "y_axis_ticks": {
            "items": y_tick_group,
            "count": len(y_tick_group),
            "spacing_px": {
                "mean": statistics.mean(y_tick_spacings) if y_tick_spacings else None,
                "std": statistics.pstdev(y_tick_spacings) if len(y_tick_spacings) > 1 else None
            }
        },
        "x_axis_ticks": {
            "items": x_tick_group,
            "count": len(x_tick_group),
            "spacing_px": {
                "mean": statistics.mean(x_tick_spacings) if x_tick_spacings else None,
                "std": statistics.pstdev(x_tick_spacings) if len(x_tick_spacings) > 1 else None
            }
        }
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result

# ============================================================
# FUNCTION 19: CHART STRUCTURE SUMMARY 
# ============================================================
def chart_structure_quality_summary(
    ocr_json_path,
    inference_json_path,
    output_json_path,
    spacing_tol_ratio=0.1,
    size_outlier_ratio=0.5,
    axis_angle_tol_deg=10.0
):
    import json
    import numpy as np
    from pathlib import Path

    ocr_json_path = Path(ocr_json_path)
    inference_json_path = Path(inference_json_path)
    output_json_path = Path(output_json_path)

    with open(ocr_json_path, "r", encoding="utf-8") as f:
        ocr_items = json.load(f)

    with open(inference_json_path, "r", encoding="utf-8") as f:
        inference = json.load(f)

    # --------------------------------------------------
    # Bars
    # --------------------------------------------------
    bars = inference.get("bars", [])
    num_bars = len(bars)
    has_bars = num_bars > 0

    # --------------------------------------------------
    # Axes
    # --------------------------------------------------
    axes = inference.get("axes", {})
    has_x_axis = "x_axis" in axes
    has_y_axis = "y_axis" in axes

    axes_perpendicular = False
    if has_x_axis and has_y_axis:
        ax = axes["x_axis"]["angle_deg"]
        ay = axes["y_axis"]["angle_deg"]
        axes_perpendicular = abs(abs(ax - ay) - 90) <= axis_angle_tol_deg

    # --------------------------------------------------
    # Identify x-axis tick labels (already correct)
    # --------------------------------------------------
    img_height = max(it["top"] + it["height"] for it in ocr_items)

    x_tick_candidates = []
    for it in ocr_items:
        txt = it["text"].strip()
        if txt.replace(".", "").isdigit():
            continue
        if it.get("orientation") == "vertical":
            continue
        y_mid = it["top"] + it["height"] / 2
        if y_mid < 0.5 * img_height:
            continue
        x_tick_candidates.append(it)

    y_centers = np.array([it["top"] + it["height"] / 2 for it in x_tick_candidates])

    x_tick_items = []
    if len(y_centers) > 0:
        median_y = np.median(y_centers)
        x_tick_items = [
            it for it in x_tick_candidates
            if abs((it["top"] + it["height"] / 2) - median_y) < it["height"]
        ]

    num_x_ticks = len(x_tick_items)

    # --------------------------------------------------
    # FIXED: Identify y-axis tick labels using inferred y-axis
    # --------------------------------------------------
    y_tick_items = []

    if has_y_axis:
        y_axis_x = axes["y_axis"]["start"][0]

        # tolerance based on image scale
        x_tol = 0.05 * max(it["left"] + it["width"] for it in ocr_items)

        for it in ocr_items:
            # must be numeric
            try:
                float(it["text"])
            except ValueError:
                continue

            # exclude vertical axis label (e.g. "USD")
            if it.get("orientation") == "vertical":
                continue

            # right edge close to y-axis line
            right_edge = it["left"] + it["width"]
            if abs(right_edge - y_axis_x) <= x_tol:
                y_tick_items.append(it)

    num_y_ticks = len(y_tick_items)


    # --------------------------------------------------
    # Tick spacing consistency (y-axis)
    # --------------------------------------------------
    y_tick_centers = sorted(
        it["top"] + it["height"] / 2 for it in y_tick_items
    )

    if len(y_tick_centers) >= 2:
        spacings = np.diff(y_tick_centers)
        spacing_mean = float(np.mean(spacings))
        spacing_std = float(np.std(spacings))
        y_spacing_consistent = (spacing_std / spacing_mean) <= spacing_tol_ratio
    else:
        spacing_mean = None
        spacing_std = None
        y_spacing_consistent = False

    def height_consistency(items, tol_ratio):
        """
        Check whether OCR boxes have consistent text height.
        """
        if len(items) < 2:
            return False, None

        heights = np.array([it["height"] for it in items])
        med = np.median(heights)
        rel_dev = np.abs(heights - med) / max(med, 1e-6)
        outliers = rel_dev > tol_ratio

        return not np.any(outliers), int(np.sum(outliers))

    x_tick_height_consistent, x_tick_height_outliers = height_consistency(
        x_tick_items, size_outlier_ratio
    )

    y_tick_height_consistent, y_tick_height_outliers = height_consistency(
        y_tick_items, size_outlier_ratio
    )


    # --------------------------------------------------
    # Title / axis labels presence
    # --------------------------------------------------
    has_title = any(it["top"] < 0.25 * img_height for it in ocr_items)
    has_x_axis_label = any(it["top"] > 0.9 * img_height for it in ocr_items)
    has_y_axis_label = any(it.get("orientation") == "vertical" for it in ocr_items)

    # --------------------------------------------------
    # Bar ↔ x-tick count match
    # --------------------------------------------------
    bars_match_x_ticks = has_bars and num_x_ticks > 0 and abs(num_bars - num_x_ticks) <= 1

    # --------------------------------------------------
    # Track mapped OCR items
    # --------------------------------------------------
    mapped_items = set()

    def item_id(it):
        # stable identity for set membership
        return (it["left"], it["top"], it["width"], it["height"], it["text"])

    # x ticks
    for it in x_tick_items:
        mapped_items.add(item_id(it))

    # y ticks
    for it in y_tick_items:
        mapped_items.add(item_id(it))

    # title
    for it in ocr_items:
        if it["top"] < 0.25 * img_height:
            mapped_items.add(item_id(it))

    # x-axis label
    for it in ocr_items:
        if it["top"] > 0.9 * img_height:
            mapped_items.add(item_id(it))

    # y-axis label
    for it in ocr_items:
        if it.get("orientation") == "vertical":
            mapped_items.add(item_id(it))

    unmapped_items = [
        it for it in ocr_items
        if item_id(it) not in mapped_items
    ]

    num_unmapped = len(unmapped_items)
    has_unmapped = num_unmapped > 0


    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------
    summary = {
        "bars": {
            "num_bars": num_bars,
            "has_bars": has_bars
        },
        "x_ticks": {
            "num_x_ticks": num_x_ticks
        },
        "y_ticks": {
            "num_y_ticks": num_y_ticks,
            "spacing_mean_px": spacing_mean,
            "spacing_std_px": spacing_std,
            "spacing_consistent": y_spacing_consistent
        },
        "text_height_consistency": {
            "x_tick_height_consistent": x_tick_height_consistent,
            "x_tick_height_outliers": x_tick_height_outliers,
            "y_tick_height_consistent": y_tick_height_consistent,
            "y_tick_height_outliers": y_tick_height_outliers
        },
        "labels": {
            "has_title": has_title,
            "has_x_axis_label": has_x_axis_label,
            "has_y_axis_label": has_y_axis_label
        },
        "axes": {
            "has_x_axis": has_x_axis,
            "has_y_axis": has_y_axis,
            "axes_perpendicular": axes_perpendicular
        },
        "cross_consistency": {
            "bars_match_x_ticks": bars_match_x_ticks
        },
        "text_mapping": {
            "num_ocr_items": len(ocr_items),
            "num_mapped": len(mapped_items),
            "num_unmapped": num_unmapped,
            "has_unmapped": has_unmapped,
            "unmapped_examples": [
                it["text"] for it in unmapped_items[:3]
            ]
        }
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
