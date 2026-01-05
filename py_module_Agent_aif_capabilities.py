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
