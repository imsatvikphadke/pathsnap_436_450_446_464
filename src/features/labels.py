"""Per-point reference labels (owner: Samyak).

.route lists the arcs the car drove, in order - it does NOT say which GPS point is on which arc.
We assign every GPS point to one route arc so that the assignment never goes backwards along the
route (monotonic alignment, solved with dynamic programming like DTW) and the total point-to-arc
distance is minimal. Points that end up far from their arc get -1 (label not trustworthy).
"""
import numpy as np

from src.common.geo import point_to_segment

MAX_LABEL_DIST = 50.0  # metres


def label_points(px, py, route, arcs, nx, ny):
    """px, py: GPS points in metres (N,); route: list of arc ids; arcs: DataFrame; nx, ny: node x/y.

    Returns (true_arc (N,), label_dist (N,)). true_arc = -1 where the aligned distance > MAX_LABEL_DIST.
    """
    route = np.asarray(route)
    f = arcs.from_node.to_numpy()[route]
    t = arcs.to_node.to_numpy()[route]
    ax, ay, bx, by = nx[f], ny[f], nx[t], ny[t]
    n, r = len(px), len(route)
    idx = np.arange(r)

    back = np.empty((n, r), dtype=np.int32)  # back[i, j] = best route index for point i-1 given point i is on j
    cost, _ = point_to_segment(px[0], py[0], ax, ay, bx, by)
    for i in range(1, n):
        # best previous cost among route indices <= j (the car cannot move backwards on the route)
        run_min = np.minimum.accumulate(cost)
        back[i] = np.maximum.accumulate(np.where(cost == run_min, idx, 0))
        d, _ = point_to_segment(px[i], py[i], ax, ay, bx, by)
        cost = run_min + d

    j = int(cost.argmin())
    path = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        path[i] = j
        j = back[i, j] if i > 0 else j

    label_dist, _ = point_to_segment(px, py, ax[path], ay[path], bx[path], by[path])
    true_arc = np.where(label_dist <= MAX_LABEL_DIST, route[path], -1)
    return true_arc, label_dist
