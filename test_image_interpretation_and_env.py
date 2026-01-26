# filename: test_image_interpretation_and_env.py

from pathlib import Path
import numpy as np

from module_aif_bar_reader.py_module_Agent_aif_capabilities import (
    image_interpretation_output_to_agent
)
from module_aif_bar_reader.env import BarChartEnv, NULL, CORRECT, INCORRECT


def main():
    # -------------------------------------------------
    # 1. Run image interpretation (perception stage)
    # -------------------------------------------------
    base_dir = Path("output_python_scripts/dir_demo_bar_chart_full_pipeline")

    axes_and_bars_path = base_dir / "inferred_axes_and_bars.json"
    ocr_path = base_dir / "ocr_data.json"

    bar_heights, tick_values, number_of_ticks, number_of_bars = (
        image_interpretation_output_to_agent(
            axes_and_bars_json_path=str(axes_and_bars_path),
            ocr_json_path=str(ocr_path),
        )
    )

    print("=== Image Interpretation Output ===")
    print(f"Bar heights (pixels): {bar_heights}")
    print(f"Tick values (pixels): {tick_values}")
    print(f"Number of ticks: {number_of_ticks}")
    print(f"Number of bars: {number_of_bars}")
    print()

    # -------------------------------------------------
    # 2. Initialize environment (generative process)
    # -------------------------------------------------
    env = BarChartEnv(
        bar_heights_pixels=bar_heights[:2],   # env currently assumes 2 bars
        tick_values_pixels=tick_values,
        n_coarse_bins=5,
        avg_tolerance=1e-6
    )

    print("=== Environment Ground Truth ===")
    print(f"True bar states: {env.true_states}")
    print(f"True average (pixels): {env.true_avg}")
    print(
        f"True average (tick, coarse): "
        f"({env.true_avg_tick}, {env.true_avg_coarse})"
    )
    print()

    # -------------------------------------------------
    # 3. Probe the environment with actions
    # -------------------------------------------------
    print("=== Environment Interaction ===")

    actions = {
        0: "bar 1 tick interval",
        1: "bar 1 coarse bin",
        2: "bar 2 tick interval",
        3: "bar 2 coarse bin",
    }

    for action, label in actions.items():
        obs = env.step(action)
        print(f"Action {action} ({label}) → obs = {obs}")

    # -------------------------------------------------
    # 4. Test report action (correct vs incorrect)
    # -------------------------------------------------
    print()

    correct_estimate = env.true_avg
    incorrect_estimate = env.true_avg + 10.0

    obs_correct = env.step(4, agent_avg_estimate=correct_estimate)
    obs_incorrect = env.step(4, agent_avg_estimate=incorrect_estimate)

    print("Report average (correct):", obs_correct)
    print("Report average (incorrect):", obs_incorrect)


if __name__ == "__main__":
    main()
