"""HMM transition model on the road graph (owner: Sanyam).

A transition = "the car was on arc a at point t-1 and is on arc b at point t".
It is plausible when the distance you would DRIVE from a to b (along the roads) is about the same as the
straight distance between the two GPS points (Newson & Krumm 2009):

    route_dist = rest of arc a  +  shortest road path (to_node(a) -> from_node(b))  +  used part of arc b
    log P(b | a) = -|route_dist - gps_dist| / beta

Same arc -> log P = 0. No road path within the search radius -> a finite -penalty (never -inf, so the
decoder can still recover when the road graph is disconnected).
"""
import networkx as nx
import numpy as np

DEFAULT_CFG = {
    "beta": 10.0,           # metres of mismatch that cost 1 unit of log-probability
    "penalty": 20.0,        # cost of an unreachable transition (finite on purpose)
    "lam": 1.0,             # weight of the emission (classifier) term
    "uturn_penalty": 5.0,   # extra cost for jumping onto the reverse arc of the same road
    "gap_s": 10.0,          # dt above this: beta is scaled by dt (more uncertainty)
    "gap_restart_s": 60.0,  # dt above this: decode the two sides of the gap independently
    "crop_m": 1000.0,       # road graph is cropped to the track bounding box + this margin
}


class RoadGraph:
    """Directed road graph in metres with cached, distance-limited Dijkstra searches."""

    def __init__(self, arcs, nx_, ny_, track_x, track_y, crop_m=1000.0):
        self.frm = arcs.from_node.to_numpy()
        self.to = arcs.to_node.to_numpy()
        self.length = np.hypot(nx_[self.to] - nx_[self.frm], ny_[self.to] - ny_[self.frm])

        # crop: only arcs near the trip matter, this keeps graph building fast on 300k-arc networks
        x0, x1 = track_x.min() - crop_m, track_x.max() + crop_m
        y0, y1 = track_y.min() - crop_m, track_y.max() + crop_m
        inside = (nx_ >= x0) & (nx_ <= x1) & (ny_ >= y0) & (ny_ <= y1)
        keep = np.flatnonzero(inside[self.frm] & inside[self.to])

        self.g = nx.DiGraph()
        self.g.add_weighted_edges_from(zip(self.frm[keep], self.to[keep], self.length[keep]))
        self._cache = {}  # source node -> (cutoff, {target: dist})

    def node_dist(self, src, dst, cutoff):
        """Shortest driving distance src -> dst in metres, or inf if not reachable within cutoff."""
        if src == dst:
            return 0.0
        hit = self._cache.get(src)
        if hit is None or hit[0] < cutoff:
            dists = nx.single_source_dijkstra_path_length(self.g, src, cutoff=cutoff) if src in self.g else {}
            hit = (cutoff, dists)
            self._cache[src] = hit
        d = hit[1].get(dst, np.inf)
        return d if d <= cutoff else np.inf


def transition_matrix(prev_arcs, prev_u, cur_arcs, cur_u, gps_dist, dt, graph, cfg):
    """log P(cur | prev) as a (len(prev), len(cur)) matrix."""
    if dt > cfg["gap_restart_s"]:
        # very long gap: forget the past -> equivalent to decoding the two segments separately
        return np.zeros((len(prev_arcs), len(cur_arcs)))

    beta = cfg["beta"] * (max(dt, 1.0) if dt > cfg["gap_s"] else 1.0)
    cutoff = min(3 * gps_dist + 200.0, 5000.0)
    L, frm, to = graph.length, graph.frm, graph.to
    A = np.full((len(prev_arcs), len(cur_arcs)), -cfg["penalty"])

    for i, (a, ua) in enumerate(zip(prev_arcs, prev_u)):
        for j, (b, ub) in enumerate(zip(cur_arcs, cur_u)):
            if a == b:
                A[i, j] = 0.0
                continue
            sp = graph.node_dist(to[a], frm[b], cutoff)
            if not np.isfinite(sp):
                continue  # unreachable -> keep -penalty
            route_dist = (1 - ua) * L[a] + sp + ub * L[b]
            A[i, j] = -abs(route_dist - gps_dist) / beta
            if frm[b] == to[a] and to[b] == frm[a]:
                A[i, j] -= cfg["uturn_penalty"]
    return A
