"""Features for every (GPS point, candidate arc) pair (owner: Samyak v1, Satvik v2)."""
import numpy as np

from src.common.geo import angle_diff, bearing_deg, point_to_segment

FEATURE_COLS = [
    "dist_m", "dist_rank", "dist_gap", "n_cands", "proj_u", "heading_diff", "speed_mps", "is_gap",
    "arc_len_m", "dist_prev", "dist_next", "in_deg", "out_deg", "two_way",
]


def _dist_to_arc(pi, arc, pts, arcs, nx, ny):
    f, t = arcs.from_node.to_numpy()[arc], arcs.to_node.to_numpy()[arc]
    d, _ = point_to_segment(pts.x.to_numpy()[pi], pts.y.to_numpy()[pi], nx[f], ny[f], nx[t], ny[t])
    return d


def make_features(pts, cand, arcs, nx, ny):
    """pts: cleaned track (clean_track); cand: DataFrame(pi, cand_arc, dist_m, proj_u) -> cand + FEATURE_COLS."""
    c = cand.copy()
    arc = c.cand_arc.to_numpy()
    pi = c.pi.to_numpy()
    f, t = arcs.from_node.to_numpy()[arc], arcs.to_node.to_numpy()[arc]

    g = c.groupby("pi").dist_m
    c["dist_rank"] = g.rank(method="first").astype(int) - 1
    c["dist_gap"] = c.dist_m - g.transform("min")
    c["n_cands"] = g.transform("size")

    # direction: does the car move the same way as the (directed) arc?
    arc_bearing = bearing_deg(nx[f], ny[f], nx[t], ny[t])
    c["heading_diff"] = angle_diff(pts.heading.to_numpy()[pi], arc_bearing)
    c["speed_mps"] = pts.speed_mps.to_numpy()[pi]
    c["is_gap"] = pts.is_gap.to_numpy()[pi].astype(int)
    c["arc_len_m"] = np.hypot(nx[t] - nx[f], ny[t] - ny[f])

    # temporal context: is the previous / next GPS point also close to this arc?
    n = len(pts)
    c["dist_prev"] = np.where(pi > 0, _dist_to_arc(np.maximum(pi - 1, 0), arc, pts, arcs, nx, ny), c.dist_m)
    c["dist_next"] = np.where(pi < n - 1, _dist_to_arc(np.minimum(pi + 1, n - 1), arc, pts, arcs, nx, ny), c.dist_m)

    # road-graph context
    in_deg = np.bincount(arcs.to_node.to_numpy(), minlength=len(nx))
    out_deg = np.bincount(arcs.from_node.to_numpy(), minlength=len(nx))
    c["in_deg"] = in_deg[f]
    c["out_deg"] = out_deg[t]
    pair = arcs.from_node.to_numpy().astype(np.int64) * len(nx) + arcs.to_node.to_numpy()
    c["two_way"] = np.isin(t.astype(np.int64) * len(nx) + f, pair).astype(int)
    return c
