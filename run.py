# filename: run.py

import numpy as np
from pathlib import Path

from pymdp.agent import Agent
import json

# -------------------------------------------------
# Imports from your module (NEW PLAN)
# -------------------------------------------------
from module_aif_bar_reader.env import (
    BarChartEnv,
    NULL,
    NOT_CLOSE_AT_ALL,
    CLOSE,
    VERY_CLOSE,
)

from module_aif_bar_reader.generate_model_ABCD_params import (
    generate_model_params,
    FINE_BINS_PER_TICK,
    COARSE_BINS_PER_TICK,
)

from module_aif_bar_reader.py_module_Agent_observation_capabilities import (
    run_bar_chart_full_pipeline
)
from module_aif_bar_reader.py_module_Agent_aif_capabilities import (
    image_interpretation_output_to_agent
)

def entropy(p, eps=1e-16):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))

def check_probvec(p, name):
    p = np.asarray(p, dtype=float)
    ok = np.all(np.isfinite(p)) and np.all(p >= -1e-6) and abs(p.sum() - 1.0) < 1e-3
    if not ok:
        raise ValueError(f"Bad probvec {name}: sum={p.sum()}, min={p.min()}, max={p.max()}, finite={np.all(np.isfinite(p))}")

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def save_probvec_png(p: np.ndarray, outpath: Path, title: str, xlabel: str = "state index", ylabel: str = "probability"):
    """Save a probability vector as a bar plot PNG."""
    p = np.asarray(p, dtype=float).ravel()
    outpath.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.bar(np.arange(len(p)), p)  # default color (don’t set)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0.0, 1.0)  # probability scale
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -------------------------------------------------
# Action semantics (NEW PLAN)
# -------------------------------------------------
ATTN_MEANINGS = {
    0: "focus_bar1",
    1: "focus_bar2",
    2: "report_avg",
}

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

def decode_coarse_symbol(sym):
    """0=NULL else coarse_global = sym-1"""
    if sym == NULL:
        return None
    return int(sym) - 1

def onehot(i, n, dtype=np.float32):
    v = np.zeros(n, dtype=dtype)
    v[int(i)] = 1.0
    return v

# -------------------------------------------------
# Policy builder (policy_len=1)
# -------------------------------------------------
def build_policies(Ncoarse, include_null_report_policy=True, debug_print=True):
    """
    Build policies for the NEW PLAN model with 4 hidden state factors:

      factor 0: bar1_fine      (uncontrolled)  -> placeholder 0 in policy column 0
      factor 1: bar2_fine      (uncontrolled)  -> placeholder 0 in policy column 1
      factor 2: attention      (controlled)    -> u_attn in {0,1,2}
      factor 3: report_choice  (controlled)    -> u_rep in {0..Ncoarse}

    policy_len = 1, so each policy is shape (1, 4).

    Semantics:
      - If u_attn in {0,1} (focus_bar1 / focus_bar2): force u_rep = 0 (NULL)
      - If u_attn == 2 (report_avg): allow u_rep in {0..Ncoarse}
          * If include_null_report_policy=False, uses only 1..Ncoarse (always non-NULL feedback)
    """
    policies = []
    n_factors = 4  # MUST match len(num_controls) and len(D)

    # ---------------------------
    # focus_bar1, report NULL
    # ---------------------------
    pol = np.zeros((1, n_factors), dtype=int)
    pol[0, 2] = 0  # attention = focus_bar1
    pol[0, 3] = 0  # report_choice = NULL
    policies.append(pol)

    # ---------------------------
    # focus_bar2, report NULL
    # ---------------------------
    pol = np.zeros((1, n_factors), dtype=int)
    pol[0, 2] = 1  # attention = focus_bar2
    pol[0, 3] = 0  # report_choice = NULL
    policies.append(pol)

    # ---------------------------
    # report_avg, choose report bin
    # ---------------------------
    start_rep = 0 if include_null_report_policy else 1
    for u_rep in range(start_rep, Ncoarse + 1):
        pol = np.zeros((1, n_factors), dtype=int)
        pol[0, 2] = 2      # attention = report_avg
        pol[0, 3] = u_rep  # report_choice
        policies.append(pol)

    # ---------------------------
    # Debug prints
    # ---------------------------
    if debug_print:
        print("First 20 policies:")
        for i, p in enumerate(policies[:20]):
            print(i, p.tolist())

        has_bar2 = any(int(p[0, 2]) == 1 for p in policies)
        has_report_nonnull = any(int(p[0, 2]) == 2 and int(p[0, 3]) > 0 for p in policies)
        has_report_null = any(int(p[0, 2]) == 2 and int(p[0, 3]) == 0 for p in policies)

        print("Has focus_bar2 policies?", has_bar2)
        print("Has report_avg with non-NULL report policies?", has_report_nonnull)
        print("Has report_avg with NULL report policy?", has_report_null)
        print("Total policies:", len(policies))

    return policies



# -------------------------------------------------
# Pretty-printing
# -------------------------------------------------
def print_beliefs(qs, env, precision=3, topk=6):
    """
    qs order matches hidden state factors:
      0 bar1_fine (Nfine)
      1 bar2_fine (Nfine)
      2 attention (3)
      3 report_choice (Ncoarse+1)
    """
    q0, q1, q2, q3 = qs
    def top_indices(q, k):
        idx = np.argsort(-q)[:k]
        return [(int(i), float(q[i])) for i in idx]

    print("Beliefs:")
    print(f"  attention: {np.round(q2, precision)}  (0=bar1,1=bar2,2=report)")
    print(f"  report_choice: top {min(topk, len(q3))} {top_indices(q3, min(topk, len(q3)))}")
    print(f"  bar1_fine: top {topk} {top_indices(q0, topk)}")
    print(f"  bar2_fine: top {topk} {top_indices(q1, topk)}")

def interpret_obs(obs):
    o0, o1, o2 = obs
    cg0 = decode_coarse_symbol(o0)
    cg1 = decode_coarse_symbol(o1)
    parts = []
    parts.append(f"o0(bar1_coarse)={ 'NULL' if cg0 is None else cg0 }")
    parts.append(f"o1(bar2_coarse)={ 'NULL' if cg1 is None else cg1 }")
    parts.append(f"o2(feedback)={feedback_to_str(o2)}")
    return ", ".join(parts)

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():

    # -------------------------------------------------
    # 1) Perception pipeline (CV + OCR)
    # -------------------------------------------------
    image_path = "input_python_scripts/two_bar_chart.png"
    output_dir = Path("output_python_scripts/dir_two_bar_chart_full_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)

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

    axes_and_bars_path = output_dir / "inferred_axes_and_bars.json"
    ocr_path = output_dir / "ocr_data.json"

    bar_values, tick_values, n_ticks, n_bars = image_interpretation_output_to_agent(
        axes_and_bars_json_path=str(axes_and_bars_path),
        ocr_json_path=str(ocr_path),
    )

    print("\n=== IMAGE INTERPRETATION OUTPUT ===")
    print(f"Bar values (data units): {bar_values}")
    print(f"Tick values (data units): {tick_values}")
    print(f"n_ticks: {n_ticks}, n_bars: {n_bars}")

    # -------------------------------------------------
    # 2) Environment (NEW PLAN)
    # -------------------------------------------------
    env = BarChartEnv(
        bar_heights_values=bar_values[:2],
        tick_values=tick_values,
        fine_bins_per_tick=FINE_BINS_PER_TICK,
        coarse_bins_per_tick=COARSE_BINS_PER_TICK,
        obs_noise_sigma=None,   # you can set e.g. 0.75 later
        seed=0,
        init_attention=0,
    )

    truth = env.get_true_states()
    print("\n=== ENV TRUE STATES (HIDDEN) ===")
    print(truth)
    print(f"Nfine={env.Nfine}, Ncoarse={env.Ncoarse}")

    # -------------------------------------------------
    # 3) Generative model ABCD (NEW PLAN)
    # -------------------------------------------------
    A, B, C, D = generate_model_params(
        n_ticks=env.n_ticks,
        fine_bins_per_tick=env.n_fine,
        coarse_bins_per_tick=env.n_coarse,
        sigma_coarse=1.0,          # likelihood noise in coarse-global space
        start_focus_bar1=True,
        dtype=np.float32,
    )

    No = [A[m].shape[0] for m in range(len(A))]  # [Ncoarse+1, Ncoarse+1, 4]

    # -------------------------------------------------
    # 4) Policies and Agent
    # -------------------------------------------------
    policies = build_policies(env.Ncoarse)

    # num_controls must align with Ns length:
    # Ns = [Nfine, Nfine, 3, Ncoarse+1] -> controls [1,1,3,Ncoarse+1]
    num_controls = [1, 1, 3, (env.Ncoarse + 1)]
    control_fac_idx = [2, 3]  # attention and report_choice

    agent = Agent(
        A=A,
        B=B,
        C=C,
        D=D,
        policies=policies,
        policy_len=1,
        inference_horizon=1,
        control_fac_idx=control_fac_idx,
        num_controls=num_controls,
        use_states_info_gain=True,
        use_utility=True,
        gamma=8.0,
        action_selection="deterministic",
    )

    print("\n=== AGENT CONFIG ===")
    print(f"Ns: {[len(d) for d in D]}")
    print(f"No: {No}")
    print(f"num_controls: {num_controls}")
    print(f"control_fac_idx: {control_fac_idx}")
    print(f"#policies: {len(policies)}")

    # -------------------------------------------------
    # 5) Active inference loop
    # -------------------------------------------------
    T = 8

    # Start with all NULL observations (0 for each modality in new plan)
    model_obs = [0, 0, 0]

    print("\n=== ACTIVE INFERENCE TRACE ===")
    print("Initial obs:", interpret_obs(model_obs))

    # Put plots in their own folder
    PLOTS_DIR = Path("output_python_scripts/posterior_plots")
    TRACE_PATH = Path("output_python_scripts/trace_run_new_plan.json")

    trace = []

    for t in range(T):
        print(f"\n--- timestep {t} ---")
        print("Observation:", interpret_obs(model_obs))

        # 1) Infer hidden states from current observation
        qs = agent.infer_states(model_obs)

        # sanity checks
        check_probvec(qs[2], "qs[2]=attention")
        check_probvec(qs[3], "qs[3]=report_choice")
        check_probvec(qs[0], "qs[0]=bar1_fine")
        check_probvec(qs[1], "qs[1]=bar2_fine")

        # 2) Infer policies
        agent.infer_policies()
        q_pi = getattr(agent, "q_pi", None)
        G = getattr(agent, "G", None)

        # 3) Choose action
        chosen = agent.sample_action()
        u_attn = int(chosen[2])
        u_rep  = int(chosen[3])
        print(f"Chosen action: u_attn={u_attn} ({ATTN_MEANINGS[u_attn]}), u_rep={u_rep}")

        # ---- clamp control-state beliefs to executed action ----
        qs[2] = onehot(u_attn, 3)
        qs[3] = onehot(u_rep, env.Ncoarse + 1)
        agent.qs = qs

        # ---- build diag AFTER clamp ----
        diag = {
            "t": int(t),
            "obs": list(map(int, model_obs)),
            "obs_decoded": {
                "bar1_cg": decode_coarse_symbol(model_obs[0]),
                "bar2_cg": decode_coarse_symbol(model_obs[1]),
                "feedback": feedback_to_str(model_obs[2]),
            },
            "Hs": {
                "H_bar1_fine": float(entropy(qs[0])),
                "H_bar2_fine": float(entropy(qs[1])),
                "H_report_choice": float(entropy(qs[3])),
            },
            "MAP": {
                "bar1_fine": int(np.argmax(qs[0])),
                "bar2_fine": int(np.argmax(qs[1])),
                "attention": int(np.argmax(qs[2])),
                "report_choice": int(np.argmax(qs[3])),
            },
            "action": {"u_attn": int(u_attn), "u_rep": int(u_rep)},
            "action_str": f"{ATTN_MEANINGS[u_attn]}, report_choice={u_rep}",
        }

        if q_pi is not None:
            q_pi = np.asarray(q_pi, dtype=float)
            diag["q_pi_top"] = [(int(i), float(q_pi[i])) for i in np.argsort(-q_pi)[:10]]

        if G is not None:
            G = np.asarray(G, dtype=float)
            diag["G_top_best"] = [(int(i), float(G[i])) for i in np.argsort(G)[:10]]

        # 4) Save posterior plots (bar1_fine and bar2_fine) to their own folders
        b1cg = diag["obs_decoded"]["bar1_cg"]
        b2cg = diag["obs_decoded"]["bar2_cg"]
        fb   = diag["obs_decoded"]["feedback"]

        save_probvec_png(
            qs[0],
            PLOTS_DIR / "bar1_fine" / f"t{t:03d}_b1cg-{b1cg}_b2cg-{b2cg}_fb-{fb}.png",
            title=f"q(bar1_fine) at t={t} | obs=({b1cg},{b2cg},{fb})"
        )
        save_probvec_png(
            qs[1],
            PLOTS_DIR / "bar2_fine" / f"t{t:03d}_b1cg-{b1cg}_b2cg-{b2cg}_fb-{fb}.png",
            title=f"q(bar2_fine) at t={t} | obs=({b1cg},{b2cg},{fb})"
        )

        # 5) Print beliefs snapshot
        print("Beliefs:")
        print_beliefs(qs, env)

        # 6) Log
        trace.append(diag)

        # 7) Step environment
        model_obs = env.step((u_attn, u_rep))

    # write json once at end
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRACE_PATH, "w") as f:
        json.dump(trace, f, indent=2)

    print(f"\nSaved trace to: {TRACE_PATH.resolve()}")
    print(f"Saved posterior plots under: {PLOTS_DIR.resolve()}")

    print("\nDone.")



if __name__ == "__main__":
    main()
