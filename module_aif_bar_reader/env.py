# filename: env.py
import numpy as np

# ============================================================
# GLOBAL DEFAULTS (bins per tick interval)
# ============================================================
FINE_BINS_PER_TICK = 10
COARSE_BINS_PER_TICK = 3

# ============================================================
# Observation encoding (0 = NULL always)
# ============================================================
NULL = 0

# avg_feedback outcomes (modality o2)
NOT_CLOSE_AT_ALL = 1
CLOSE = 2
VERY_CLOSE = 3


class BarChartEnv:
    """
    Generative process for the START-OVER MODEL SPEC (new plan).

    Hidden state factors (ground truth + controlled internal):
      s0: bar1_fine       in {0..Nfine-1}      (fixed true state)
      s1: bar2_fine       in {0..Nfine-1}      (fixed true state)
      s2: attention       in {0,1,2}           (controlled; set by u_attn)
      s3: report_choice   in {0..Ncoarse}      (controlled; set by u_rep)
          0 = NULL (no report selected)
          1..Ncoarse = reported_avg_coarse_global_bin + 1

    Observation modalities:
      o0: bar1_coarse_obs in {0..Ncoarse}      (0 NULL, else coarse_global+1)
      o1: bar2_coarse_obs in {0..Ncoarse}
      o2: avg_feedback    in {0,1,2,3}         (0 NULL, 1 not_close, 2 close, 3 very_close)

    step(action) expects:
      action = (u_attn, u_rep) or [u_attn, u_rep]
        u_attn in {0,1,2}
        u_rep  in {0..Ncoarse}   (0 means NULL report; k means choose report_choice=k)

    Returns:
      obs = [o0, o1, o2]
    """

    def __init__(
        self,
        bar_heights_values,             # [v1, v2] in data units
        tick_values,                    # ascending list, length n_ticks+1 (will be sorted)
        fine_bins_per_tick=FINE_BINS_PER_TICK,
        coarse_bins_per_tick=COARSE_BINS_PER_TICK,
        obs_noise_sigma=None,           # std dev in "global coarse bins" for noisy sampling
        seed=None,
        init_attention=0,               # 0 focus_bar1, 1 focus_bar2, 2 report_avg
    ):
        if len(bar_heights_values) != 2:
            raise ValueError("Exactly two bars required")

        self.bar_values = [float(bar_heights_values[0]), float(bar_heights_values[1])]
        self.tick_values = np.array(sorted(tick_values), dtype=float)

        if self.tick_values.ndim != 1 or len(self.tick_values) < 2:
            raise ValueError("tick_values must be a 1D list/array of length >= 2")

        self.n_ticks = len(self.tick_values) - 1
        self.n_fine = int(fine_bins_per_tick)
        self.n_coarse = int(coarse_bins_per_tick)

        if self.n_fine <= 0 or self.n_coarse <= 0:
            raise ValueError("fine_bins_per_tick and coarse_bins_per_tick must be positive")

        if self.n_coarse > self.n_fine:
            raise ValueError("coarse_bins_per_tick should be <= fine_bins_per_tick")

        self.Nfine = self.n_ticks * self.n_fine
        self.Ncoarse = self.n_ticks * self.n_coarse

        self.obs_noise_sigma = None if obs_noise_sigma is None else float(obs_noise_sigma)
        self.rng = np.random.default_rng(seed)

        # True (fixed) world hidden states: global fine indices
        self.bar1_fine_true = self._value_to_fine_global(self.bar_values[0])
        self.bar2_fine_true = self._value_to_fine_global(self.bar_values[1])

        # True average coarse global bin (0-indexed in 0..Ncoarse-1), used for feedback
        self.true_avg_coarse_global = self._true_avg_coarse_global_from_values()

        # Controlled internal states
        self.attention = int(init_attention)  # s2
        if self.attention not in (0, 1, 2):
            raise ValueError("init_attention must be 0, 1, or 2")
        self.report_choice = 0  # s3 starts at NULL

    # =========================================================
    # Discretization helpers
    # =========================================================
    def _value_to_tick_idx(self, value):
        """
        Map a value to tick interval index in 0..n_ticks-1.
        Clipped to bounds if value is outside tick range.
        """
        v = float(value)
        idx = int(np.searchsorted(self.tick_values, v, side="right") - 1)
        return int(np.clip(idx, 0, self.n_ticks - 1))

    def _value_to_fine_within(self, value, tick_idx):
        """
        Map a value to a fine sub-bin index within a tick interval: 0..n_fine-1.
        """
        lo = self.tick_values[tick_idx]
        hi = self.tick_values[tick_idx + 1]
        if hi <= lo:
            frac = 0.0
        else:
            frac = (float(value) - lo) / (hi - lo)

        frac = float(np.clip(frac, 0.0, 1.0))
        fine_within = int(np.floor(frac * self.n_fine))
        return int(np.clip(fine_within, 0, self.n_fine - 1))

    def _value_to_fine_global(self, value):
        """
        Map a data value -> global fine bin index in 0..Nfine-1.

        global_fine = tick_idx * n_fine + fine_within
        """
        tick_idx = self._value_to_tick_idx(value)
        fine_within = self._value_to_fine_within(value, tick_idx)
        return tick_idx * self.n_fine + fine_within

    def _fine_global_to_coarse_global(self, fine_global):
        """
        Map global fine bin -> global coarse bin (both 0-indexed).

        Spec mapping:
          tick        = fine_idx // n_fine
          fine_within = fine_idx %  n_fine
          coarse_within = floor( fine_within / n_fine * n_coarse )
          coarse_global = tick * n_coarse + coarse_within
        """
        fg = int(fine_global)
        tick = fg // self.n_fine
        fine_within = fg % self.n_fine

        coarse_within = (fine_within * self.n_coarse) // self.n_fine  # floor
        coarse_within = int(np.clip(coarse_within, 0, self.n_coarse - 1))

        coarse_global = tick * self.n_coarse + coarse_within
        return int(np.clip(coarse_global, 0, self.Ncoarse - 1))

    def _true_avg_coarse_global_from_values(self):
        """
        Deterministic mapping from (bar1_value, bar2_value) -> true_avg_coarse_global (0-indexed).
        Uses average value -> fine_global -> coarse_global.
        """
        avg_val = 0.5 * (self.bar_values[0] + self.bar_values[1])
        avg_fine_global = self._value_to_fine_global(avg_val)
        return self._fine_global_to_coarse_global(avg_fine_global)

    # =========================================================
    # Optional noisy sampling in coarse space
    # =========================================================
    def _sample_noisy_coarse_global(self, center_coarse_global):
        """
        If obs_noise_sigma is None or <= 0 -> deterministic center.
        Else sample from a discrete Gaussian over 0..Ncoarse-1 centered at center.
        """
        c = int(center_coarse_global)
        sig = self.obs_noise_sigma
        if sig is None or sig <= 0.0:
            return c

        xs = np.arange(self.Ncoarse, dtype=float)
        logits = -0.5 * ((xs - c) / sig) ** 2
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs /= probs.sum()
        return int(self.rng.choice(self.Ncoarse, p=probs))

    # =========================================================
    # Interaction
    # =========================================================
    def reset(self, init_attention=0):
        """
        Reset controlled states; keep same true bars.
        Returns initial observation (typically all NULL).
        """
        self.attention = int(init_attention)
        if self.attention not in (0, 1, 2):
            raise ValueError("init_attention must be 0, 1, or 2")
        self.report_choice = 0
        return [NULL, NULL, NULL]

    def step(self, action):
        """
        Apply (u_attn, u_rep), update controlled states, and emit observations [o0, o1, o2]
        according to gating rules in the spec.
        """
        if not (isinstance(action, (list, tuple)) and len(action) == 2):
            raise ValueError("action must be a length-2 (u_attn, u_rep) list/tuple")

        u_attn = int(action[0])
        u_rep = int(action[1])

        if u_attn not in (0, 1, 2):
            raise ValueError("u_attn must be 0, 1, or 2")
        if not (0 <= u_rep <= self.Ncoarse):
            raise ValueError(f"u_rep must be in 0..{self.Ncoarse}")

        # Controlled transitions (set-actions)
        self.attention = u_attn
        self.report_choice = u_rep  # 0 means NULL, else 1..Ncoarse

        # Observations default to NULL
        o0 = NULL
        o1 = NULL
        o2 = NULL

        # o0: bar1_coarse_obs (gated by attention == focus_bar1)
        if self.attention == 0:
            center = self._fine_global_to_coarse_global(self.bar1_fine_true)
            coarse_global = self._sample_noisy_coarse_global(center)
            o0 = coarse_global + 1  # reserve 0 for NULL

        # o1: bar2_coarse_obs (gated by attention == focus_bar2)
        if self.attention == 1:
            center = self._fine_global_to_coarse_global(self.bar2_fine_true)
            coarse_global = self._sample_noisy_coarse_global(center)
            o1 = coarse_global + 1

        # o2: avg_feedback (gated by attention == report_avg AND report_choice != 0)
        if self.attention == 2 and self.report_choice != 0:
            report_bin = self.report_choice - 1  # 0..Ncoarse-1
            true_bin = self.true_avg_coarse_global

            tick_true = true_bin // self.n_coarse
            tick_rep = report_bin // self.n_coarse
            coarse_true = true_bin % self.n_coarse
            coarse_rep = report_bin % self.n_coarse

            if tick_rep != tick_true:
                o2 = NOT_CLOSE_AT_ALL
            elif coarse_rep == coarse_true:
                o2 = VERY_CLOSE
            else:
                o2 = CLOSE

        return [o0, o1, o2]

    # =========================================================
    # Convenience getters (optional)
    # =========================================================
    def get_true_states(self):
        """Returns true world states (fine globals) and true avg coarse global."""
        return {
            "bar1_fine_true": int(self.bar1_fine_true),
            "bar2_fine_true": int(self.bar2_fine_true),
            "true_avg_coarse_global": int(self.true_avg_coarse_global),
        }

    def get_control_states(self):
        """Returns current controlled states (attention, report_choice)."""
        return {
            "attention": int(self.attention),
            "report_choice": int(self.report_choice),
        }
