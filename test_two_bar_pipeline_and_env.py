# filename: test_two_bar_pipeline_and_env.py

from pathlib import Path
import numpy as np

from module_aif_bar_reader.py_module_Agent_observation_capabilities import (
    run_bar_chart_full_pipeline
)
from module_aif_bar_reader.py_module_Agent_aif_capabilities import (
    image_interpretation_output_to_agent
)
from module_aif_bar_reader.env import (
    BarChartEnv,
    NULL,
    NOT_CLOSE_AT_ALL,
    CLOSE,
    VERY_CLOSE,
)


def decode_coarse_symbol(symbol):
    """
    Observation symbol -> coarse_global bin (0-indexed) or None if NULL.
    Per spec: 0=NULL, 1..Ncoarse => coarse_global = symbol-1
    """
    if symbol == NULL:
        return None
    return int(symbol) - 1


def feedback_to_str(fb):
    if fb == NULL:
        return "NULL"
    if fb == NOT_CLOSE_AT_ALL:
        return "not_close_at_all"
    if fb == CLOSE:
        return "close"
    if fb == VERY_CLOSE:
        return "very_close"
    return f"UNKNOWN({fb})"


def main():
    # -------------------------------------------------
    # Paths
    # -------------------------------------------------
    image_path = "input_python_scripts/two_bar_chart.png"
    output_dir = Path("output_python_scripts/dir_two_bar_chart_full_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # 1) Run full perception pipeline
    # -------------------------------------------------
    print("=== RUNNING FULL BAR-CHART PIPELINE ===")

    run_bar_chart_full_pipeline(
        image_path=image_path,
        output_dir=str(output_dir),
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

    print("JSON outputs written to:")
    print(output_dir.resolve())
    print()

    # -------------------------------------------------
    # 2) Perception -> agent quantities
    # -------------------------------------------------
    axes_and_bars_path = output_dir / "inferred_axes_and_bars.json"
    ocr_path = output_dir / "ocr_data.json"

    bar_values, tick_values, n_ticks, n_bars = image_interpretation_output_to_agent(
        axes_and_bars_json_path=str(axes_and_bars_path),
        ocr_json_path=str(ocr_path),
    )

    print("=== IMAGE INTERPRETATION OUTPUT ===")
    print(f"Bar values (data units): {bar_values}")
    print(f"Tick values (data units): {tick_values}")
    print(f"Number of ticks: {n_ticks}")
    print(f"Number of bars:  {n_bars}")
    print()

    # -------------------------------------------------
    # 3) Initialize environment (NEW PLAN API)
    # -------------------------------------------------
    env = BarChartEnv(
        bar_heights_values=bar_values[:2],
        tick_values=tick_values,
        obs_noise_sigma=None,
        seed=0,
        init_attention=0,
    )


    print("=== ENVIRONMENT GROUND TRUTH (HIDDEN) ===")
    truth = env.get_true_states()
    print("bar1_fine_true (global fine idx):", truth["bar1_fine_true"])
    print("bar2_fine_true (global fine idx):", truth["bar2_fine_true"])
    print("true_avg_coarse_global (0-index):", truth["true_avg_coarse_global"])
    print()

    # -------------------------------------------------
    # 4) Probe observations (attention gating)
    #    action = (u_attn, u_rep)
    #      u_attn: 0 focus_bar1, 1 focus_bar2, 2 report_avg
    #      u_rep : 0..Ncoarse  (0=NULL report)
    # -------------------------------------------------
    print("=== ENVIRONMENT INTERACTION (GATED OBS) ===")

    # Keep report NULL while probing bars
    UREP_NULL = 0

    probes = [
        ((0, UREP_NULL), "focus_bar1 (expect o0 informative, o1/o2 NULL)"),
        ((1, UREP_NULL), "focus_bar2 (expect o1 informative, o0/o2 NULL)"),
        ((2, UREP_NULL), "report_avg with NULL report (expect all NULL)"),
    ]

    for (u_attn, u_rep), label in probes:
        obs = env.step((u_attn, u_rep))
        o0, o1, o2 = obs

        print(f"\nAction (u_attn={u_attn}, u_rep={u_rep}) -> {label}")
        print(f"  Observation [o0, o1, o2]: {obs}")

        cg0 = decode_coarse_symbol(o0)
        cg1 = decode_coarse_symbol(o1)

        print(f"  o0 (bar1_coarse_obs): {'NULL' if cg0 is None else f'coarse_global={cg0}'}")
        print(f"  o1 (bar2_coarse_obs): {'NULL' if cg1 is None else f'coarse_global={cg1}'}")
        print(f"  o2 (avg_feedback):    {feedback_to_str(o2)}")

    # -------------------------------------------------
    # 5) Report average: choose a report_choice and attend to report_avg
    #    u_rep encodes report_choice directly:
    #      u_rep=0 => NULL
    #      u_rep=k => report_choice=k => reported coarse_global = k-1
    # -------------------------------------------------
    print("\n=== REPORT AVG (FEEDBACK) ===")

    Ncoarse = env.Ncoarse
    true_avg_coarse_global = truth["true_avg_coarse_global"]

    # Correct report: u_rep = true_bin + 1
    u_rep_correct = int(true_avg_coarse_global + 1)

    # Bad report: shift by one tick if possible, else shift by +n_coarse within range
    bad_bin = true_avg_coarse_global + env.n_coarse
    if bad_bin >= Ncoarse:
        bad_bin = max(0, true_avg_coarse_global - env.n_coarse)
    u_rep_bad = int(bad_bin + 1)

    obs_correct = env.step((2, u_rep_correct))
    obs_bad = env.step((2, u_rep_bad))

    print(f"True avg coarse_global: {true_avg_coarse_global} (0-indexed)")
    print(f"Report (correct) coarse_global={u_rep_correct-1} -> obs={obs_correct} -> {feedback_to_str(obs_correct[2])}")
    print(f"Report (bad)     coarse_global={u_rep_bad-1} -> obs={obs_bad} -> {feedback_to_str(obs_bad[2])}")


if __name__ == "__main__":
    main()
