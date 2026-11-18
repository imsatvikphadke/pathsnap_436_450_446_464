"""Viterbi decoding over candidate arcs (owner: Sanyam).

Hidden state at point t   = which of its candidate arcs the car is on.
Emission  log B_t(j)      = log of the classifier probability of candidate j, renormalised over the point's candidates.
Transition log A(i, j)    = src.hmm.transitions.transition_matrix (road-graph distance vs GPS distance).

Recurrence (log space, so long products never underflow):
    delta_0(j) = lam * log B_0(j)
    delta_t(j) = max_i [ delta_{t-1}(i) + log A(i, j) ] + lam * log B_t(j)
    back_t(j)  = argmax_i [ ... ]
Then backtrack from argmax_j delta_T(j).
"""
import numpy as np

from src.hmm.transitions import transition_matrix

EPS = 1e-6


def emission_logprob(prob):
    p = np.asarray(prob, dtype=float) + EPS
    return np.log(p / p.sum())


def viterbi_decode(x, y, t, cands, graph, cfg):
    """x, y, t: per-point arrays (metres, seconds). cands: list (one per point) of dicts with
    'arcs', 'u', 'prob' arrays (may be empty). Returns one arc id per point (-1 only if no point has candidates)."""
    n = len(cands)
    has = [i for i in range(n) if len(cands[i]["arcs"])]
    pred = np.full(n, -1, dtype=np.int64)
    if not has:
        return pred

    lam = cfg["lam"]
    first = has[0]
    delta = lam * emission_logprob(cands[first]["prob"])
    backs = []
    prev = first
    for i in has[1:]:
        gps_dist = float(np.hypot(x[i] - x[prev], y[i] - y[prev]))
        A = transition_matrix(cands[prev]["arcs"], cands[prev]["u"], cands[i]["arcs"], cands[i]["u"],
                              gps_dist, float(t[i] - t[prev]), graph, cfg)
        scores = delta[:, None] + A
        backs.append(scores.argmax(axis=0))
        delta = scores.max(axis=0) + lam * emission_logprob(cands[i]["prob"])
        prev = i

    # backtrack: index of the chosen candidate at each point that had candidates
    j = int(delta.argmax())
    chosen = [j]
    for b in reversed(backs):
        j = int(b[j])
        chosen.append(j)
    chosen.reverse()
    for i, j in zip(has, chosen):
        pred[i] = cands[i]["arcs"][j]

    # points with no candidates copy the nearest decoded neighbour
    if len(has) < n:
        has_arr = np.array(has)
        for i in range(n):
            if pred[i] < 0:
                pred[i] = pred[has_arr[np.abs(has_arr - i).argmin()]]
    return pred
