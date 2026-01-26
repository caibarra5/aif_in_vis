# filename: agent_env_interface.py

import numpy as np


def expected_bar_value_from_beliefs(tick_q, fine_q, env):
    """
    Compute expected bar value from beliefs over (tick, fine_bin).
    """
    estimate = 0.0

    for t_idx, pt in enumerate(tick_q):
        if pt == 0.0:
            continue
        tick_interval = t_idx + 1

        for fine_bin, pf in enumerate(fine_q):
            if pf == 0.0:
                continue

            lo, hi = env._hidden_to_interval(tick_interval, fine_bin)
            midpoint = 0.5 * (lo + hi)

            estimate += pt * pf * midpoint

    return estimate


def compute_avg_report_from_beliefs(qs, env):
    """
    Compute agent's reported average from beliefs over bar states.

    qs indexing:
        qs[0] = bar1_tick
        qs[1] = bar1_fine
        qs[2] = bar2_tick
        qs[3] = bar2_fine
    """

    bar1_est = expected_bar_value_from_beliefs(
        qs[0], qs[1], env
    )

    bar2_est = expected_bar_value_from_beliefs(
        qs[2], qs[3], env
    )

    return 0.5 * (bar1_est + bar2_est)


def env_step_with_agent(env, agent, action, qs=None):
    """
    Unified agent–environment interaction helper.

    Parameters
    ----------
    env : BarChartEnv
    agent : pymdp.Agent
    action : int
        Discrete action index sampled by agent
    qs : list[np.ndarray] | None
        Posterior beliefs (required only for action 4)

    Returns
    -------
    observation : list[int]
    """

    if action != 4:
        return env.step(action)

    # Action 4: report average
    if qs is None:
        raise ValueError("Beliefs qs required to report average")

    avg_estimate = compute_avg_report_from_beliefs(qs, env)
    return env.step(4, agent_avg_estimate=avg_estimate)
