# filename: py_module_Agent_aif_capabilities.py

import math
import json
from pathlib import Path
from itertools import combinations


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


import numpy as np
import re

import json
import re
import numpy as np


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
        if re.fullmatch(r"\d+", text):
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



def environment(bar_heights, tick_values):

    return env

def agent(number_of_ticks, number_of_bars):

    return agent