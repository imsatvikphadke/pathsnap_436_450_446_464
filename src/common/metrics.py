"""Metric definitions shared by the whole team, so every table in the report means the same thing."""
import numpy as np


def top1_accuracy(pred_arcs, true_arcs):
    """Observation-level Top-1: share of labelled GPS points whose predicted arc == reference arc."""
    pred, true = np.asarray(pred_arcs), np.asarray(true_arcs)
    mask = true >= 0  # -1 = point has no reference label
    return float((pred[mask] == true[mask]).mean()) if mask.any() else float("nan")


def candidate_miss_rate(true_arcs, candidate_sets):
    """Share of labelled points whose reference arc is NOT in that point's candidate set."""
    misses = [t not in set(c) for t, c in zip(true_arcs, candidate_sets) if t >= 0]
    return float(np.mean(misses)) if misses else float("nan")


def route_continuity(pred_arcs, arcs_df):
    """Share of consecutive arc CHANGES in the predicted sequence that are physically connected
    (to_node of previous arc == from_node of next arc). 1.0 = a drivable route."""
    seq = [a for i, a in enumerate(pred_arcs) if a >= 0 and (i == 0 or a != pred_arcs[i - 1])]
    if len(seq) < 2:
        return 1.0
    to_node = arcs_df["to_node"].to_numpy()
    from_node = arcs_df["from_node"].to_numpy()
    ok = [to_node[a] == from_node[b] for a, b in zip(seq[:-1], seq[1:])]
    return float(np.mean(ok))
