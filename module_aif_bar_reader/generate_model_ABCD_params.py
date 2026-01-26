# filename: generate_model_ABCD_params.py
import numpy as np

# ============================================================
# Defaults (MUST match env.py)
# ============================================================
FINE_BINS_PER_TICK = 10
COARSE_BINS_PER_TICK = 3

# ============================================================
# Helpers
# ============================================================
def normalize(x, axis=0, eps=1e-16):
    s = x.sum(axis=axis, keepdims=True)
    s = np.where(s < eps, 1.0, s)
    return x / s


def fine_global_to_coarse_global(fine_idx, n_fine, n_coarse):
    """
    Spec mapping:
      tick        = fine_idx // n_fine
      fine_within = fine_idx %  n_fine
      coarse_within = floor( fine_within / n_fine * n_coarse )
      coarse_global = tick * n_coarse + coarse_within
    """
    fine_idx = int(fine_idx)
    tick = fine_idx // n_fine
    fine_within = fine_idx % n_fine
    coarse_within = (fine_within * n_coarse) // n_fine
    coarse_within = int(np.clip(coarse_within, 0, n_coarse - 1))
    return tick * n_coarse + coarse_within


def discrete_trunc_gaussian_probs(center, K, sigma):
    """
    Discrete Gaussian over {0..K-1}, centered at 'center', truncated and normalized.
    If sigma is None or <= 0 -> delta at center.
    """
    center = int(center)
    if sigma is None or sigma <= 0:
        p = np.zeros(K, dtype=float)
        p[center] = 1.0
        return p

    xs = np.arange(K, dtype=float)
    logits = -0.5 * ((xs - center) / float(sigma)) ** 2
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    return p


# ============================================================
# Dimensions (NEW PLAN)
# ============================================================
def get_dimensions(n_ticks, n_fine_per_tick, n_coarse_per_tick):
    """
    Hidden state factors (4):
      s0: bar1_fine       (Nfine)       0..Nfine-1
      s1: bar2_fine       (Nfine)       0..Nfine-1
      s2: attention       (3)           0=focus_bar1,1=focus_bar2,2=report_avg
      s3: report_choice   (Ncoarse+1)   0=NULL, 1..Ncoarse => report_bin = idx-1

    Observation modalities (3):
      o0: bar1_coarse_obs (Ncoarse+1)   0=NULL, 1..Ncoarse => coarse_global=o-1
      o1: bar2_coarse_obs (Ncoarse+1)
      o2: avg_feedback    (4)           0=NULL, 1=not_close, 2=close, 3=very_close
    """
    n_ticks = int(n_ticks)
    n_fine = int(n_fine_per_tick)
    n_coarse = int(n_coarse_per_tick)

    Nfine = n_ticks * n_fine
    Ncoarse = n_ticks * n_coarse

    Ns = [Nfine, Nfine, 3, (Ncoarse + 1)]
    No = [(Ncoarse + 1), (Ncoarse + 1), 4]

    return Ns, No, Nfine, Ncoarse


# ============================================================
# A matrices (likelihood) -- NEW PLAN
# ============================================================
def build_A(n_ticks, n_fine_per_tick, n_coarse_per_tick, sigma_coarse=1.0, dtype=np.float32):
    """
    A = [A0, A1, A2]

    A0: bar1_coarse_obs
      - if attention != 0 -> NULL w.p.1
      - if attention == 0 -> truncated discrete Gaussian over coarse_global (then +1 symbol)

    A1: bar2_coarse_obs
      - if attention != 1 -> NULL w.p.1
      - if attention == 1 -> truncated discrete Gaussian over coarse_global (then +1 symbol)

    A2: avg_feedback
      - if attention != 2 -> NULL w.p.1
      - if attention == 2 and report_choice==0 -> NULL w.p.1
      - else compare report_bin to true_avg_coarse_global implied by (bar1_fine, bar2_fine)
        using your tick/within rules.
    """
    Ns, No, Nfine, Ncoarse = get_dimensions(n_ticks, n_fine_per_tick, n_coarse_per_tick)
    No0, No1, No2 = No

    n_fine = int(n_fine_per_tick)
    n_coarse = int(n_coarse_per_tick)

    # Shapes:
    # A0: (Ncoarse+1, Nfine, Nfine, 3, Ncoarse+1)
    # A1: same
    # A2: (4,        Nfine, Nfine, 3, Ncoarse+1)
    A0 = np.zeros((No0, Nfine, Nfine, 3, Ncoarse + 1), dtype=dtype)
    A1 = np.zeros((No1, Nfine, Nfine, 3, Ncoarse + 1), dtype=dtype)
    A2 = np.zeros((No2, Nfine, Nfine, 3, Ncoarse + 1), dtype=dtype)

    # -------------------------
    # A0 gating default: NULL everywhere
    # -------------------------
    A0[0, :, :, :, :] = 1.0
    # Fill attention==0 slices
    for bar1_fine in range(Nfine):
        center_cg = fine_global_to_coarse_global(bar1_fine, n_fine, n_coarse)  # 0..Ncoarse-1
        p = discrete_trunc_gaussian_probs(center_cg, Ncoarse, sigma_coarse).astype(dtype)

        # outcomes 1..Ncoarse correspond to coarse_global 0..Ncoarse-1
        A0[:, bar1_fine, :, 0, :] = 0.0
        A0[1:, bar1_fine, :, 0, :] = p[:, None, None]  # broadcast over bar2_fine and report_choice
    A0 = normalize(A0, axis=0).astype(dtype)

    # -------------------------
    # A1 gating default: NULL everywhere
    # -------------------------
    A1[0, :, :, :, :] = 1.0
    # Fill attention==1 slices
    for bar2_fine in range(Nfine):
        center_cg = fine_global_to_coarse_global(bar2_fine, n_fine, n_coarse)
        p = discrete_trunc_gaussian_probs(center_cg, Ncoarse, sigma_coarse).astype(dtype)

        A1[:, :, bar2_fine, 1, :] = 0.0
        A1[1:, :, bar2_fine, 1, :] = p[:, None, None]  # broadcast over bar1_fine and report_choice
    A1 = normalize(A1, axis=0).astype(dtype)

    # -------------------------
    # A2 gating default: NULL everywhere
    # -------------------------
    A2[0, :, :, :, :] = 1.0

    # Only define attention==2 slices.
    # Deterministic mapping:
    #   (bar1_fine, bar2_fine) -> true_avg_coarse_global
    # We approximate "avg implied by fine indices" using average of fine_global indices.
    # (Keeps this generator independent of tick_values.)
    for bar1_fine in range(Nfine):
        for bar2_fine in range(Nfine):
            avg_fine = int(np.floor(0.5 * (bar1_fine + bar2_fine)))
            true_cg = fine_global_to_coarse_global(avg_fine, n_fine, n_coarse)

            tick_true = true_cg // n_coarse
            coarse_true = true_cg % n_coarse

            for rep_choice in range(Ncoarse + 1):
                if rep_choice == 0:
                    # NULL report -> NULL feedback
                    A2[:, bar1_fine, bar2_fine, 2, rep_choice] = 0.0
                    A2[0, bar1_fine, bar2_fine, 2, rep_choice] = 1.0
                else:
                    report_bin = rep_choice - 1
                    tick_rep = report_bin // n_coarse
                    coarse_rep = report_bin % n_coarse

                    if tick_rep != tick_true:
                        fb = 1  # not_close_at_all
                    elif coarse_rep == coarse_true:
                        fb = 3  # very_close
                    else:
                        fb = 2  # close

                    A2[:, bar1_fine, bar2_fine, 2, rep_choice] = 0.0
                    A2[fb, bar1_fine, bar2_fine, 2, rep_choice] = 1.0

    A2 = normalize(A2, axis=0).astype(dtype)

    return np.array([A0, A1, A2], dtype=object)


# ============================================================
# B matrices (transitions) -- NEW PLAN
# ============================================================
def build_B_identity(n, dtype=np.float32):
    B = np.zeros((n, n, 1), dtype=dtype)
    np.fill_diagonal(B[:, :, 0], 1.0)
    return B

def build_B(n_ticks, n_fine_per_tick, n_coarse_per_tick, dtype=np.float32):
    _, _, Nfine, Ncoarse = get_dimensions(n_ticks, n_fine_per_tick, n_coarse_per_tick)

    # B0: bar1_fine identity
    B0 = build_B_identity(Nfine, dtype=dtype)

    # B1: bar2_fine identity
    B1 = build_B_identity(Nfine, dtype=dtype)

    # B2: attention set-action (3 states, 3 actions)
    B2 = np.zeros((3, 3, 3), dtype=dtype)
    for u in range(3):
        B2[u, :, u] = 1.0

    # B3: report_choice set-action (Ncoarse+1 states, Ncoarse+1 actions)
    B3 = np.zeros((Ncoarse + 1, Ncoarse + 1, Ncoarse + 1), dtype=dtype)
    for u in range(Ncoarse + 1):
        B3[u, :, u] = 1.0

    return np.array([B0, B1, B2, B3], dtype=object)


# ============================================================
# C preferences -- NEW PLAN
# ============================================================
def build_C(n_ticks, n_fine_per_tick, n_coarse_per_tick, dtype=np.float32):
    _, No, _, _ = get_dimensions(n_ticks, n_fine_per_tick, n_coarse_per_tick)
    C = [np.zeros(n, dtype=dtype) for n in No]

    # avg_feedback: [NULL, not_close, close, very_close]
    C[2][0] = 0.0
    C[2][1] = -6.0
    C[2][2] = +1.0
    C[2][3] = +6.0

    return np.array(C, dtype=object)


# ============================================================
# D priors -- NEW PLAN
# ============================================================
def build_D(n_ticks, n_fine_per_tick, n_coarse_per_tick, start_focus_bar1=True, dtype=np.float32):
    _, _, Nfine, Ncoarse = get_dimensions(n_ticks, n_fine_per_tick, n_coarse_per_tick)

    D0 = (np.ones(Nfine, dtype=dtype) / Nfine).astype(dtype)
    D1 = (np.ones(Nfine, dtype=dtype) / Nfine).astype(dtype)

    if start_focus_bar1:
        D2 = np.array([1.0, 0.0, 0.0], dtype=dtype)
    else:
        D2 = (np.ones(3, dtype=dtype) / 3.0).astype(dtype)

    D3 = np.zeros(Ncoarse + 1, dtype=dtype)
    D3[0] = 1.0  # start with NULL report_choice

    return np.array([D0, D1, D2, D3], dtype=object)


# ============================================================
# Entry point
# ============================================================
def generate_model_params(
    n_ticks=5,
    fine_bins_per_tick=FINE_BINS_PER_TICK,
    coarse_bins_per_tick=COARSE_BINS_PER_TICK,
    sigma_coarse=1.0,
    start_focus_bar1=True,
    dtype=np.float32,
):
    A = build_A(
        n_ticks=n_ticks,
        n_fine_per_tick=fine_bins_per_tick,
        n_coarse_per_tick=coarse_bins_per_tick,
        sigma_coarse=sigma_coarse,
        dtype=dtype,
    )
    B = build_B(
        n_ticks=n_ticks,
        n_fine_per_tick=fine_bins_per_tick,
        n_coarse_per_tick=coarse_bins_per_tick,
        dtype=dtype,
    )
    C = build_C(
        n_ticks=n_ticks,
        n_fine_per_tick=fine_bins_per_tick,
        n_coarse_per_tick=coarse_bins_per_tick,
        dtype=dtype,
    )
    D = build_D(
        n_ticks=n_ticks,
        n_fine_per_tick=fine_bins_per_tick,
        n_coarse_per_tick=coarse_bins_per_tick,
        start_focus_bar1=start_focus_bar1,
        dtype=dtype,
    )

    return A, B, C, D


if __name__ == "__main__":
    A, B, C, D = generate_model_params()
    print("ABCD parameters generated (NEW PLAN).")
