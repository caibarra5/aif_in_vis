from pathlib import Path
from py_module_Agent_aif_capabilities import image_interpretation_output_to_agent


def main():
    base_dir = Path("dir_demo_bar_chart_full_pipeline")

    axes_and_bars_path = base_dir / "inferred_axes_and_bars.json"
    ocr_path = base_dir / "ocr_data.json"

    bar_heights, tick_values, number_of_ticks, number_of_bars = (
        image_interpretation_output_to_agent(
            axes_and_bars_json_path=str(axes_and_bars_path),
            ocr_json_path=str(ocr_path),
        )
    )

    print("=== Image Interpretation Output ===")
    print(f"Bar heights (data units): {bar_heights}")
    print(f"Tick values: {tick_values}")
    print(f"Number of ticks: {number_of_ticks}")
    print(f"Number of bars: {number_of_bars}")


if __name__ == "__main__":
    main()
