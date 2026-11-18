"""Candidate road arcs for every GPS point (owner: Samyak).

We cannot score every arc of a 166k-arc network for every point, so each point only considers the
K closest arcs within radius R. Arcs are sampled every SAMPLE_M metres and put in a KD-tree, so
"which arcs are near this point" is a fast tree lookup instead of a scan.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.common.geo import point_to_segment

RADIUS_M = 50.0
TOP_K = 8
SAMPLE_M = 10.0
CROP_M = 300.0
FALLBACK_K = 3  # if nothing is within R, keep the 3 closest arcs anyway (never an empty set)


def crop_arcs(px, py, arcs, nx, ny, margin=CROP_M):
    """Arc ids whose bounding box overlaps the track bounding box + margin."""
    fx, fy = nx[arcs.from_node.to_numpy()], ny[arcs.from_node.to_numpy()]
    tx, ty = nx[arcs.to_node.to_numpy()], ny[arcs.to_node.to_numpy()]
    x0, x1, y0, y1 = px.min() - margin, px.max() + margin, py.min() - margin, py.max() + margin
    keep = (np.maximum(fx, tx) >= x0) & (np.minimum(fx, tx) <= x1) & (np.maximum(fy, ty) >= y0) & (np.minimum(fy, ty) <= y1)
    return arcs.index.to_numpy()[keep]


def _arc_samples(arc_ids, arcs, nx, ny):
    f, t = arcs.from_node.to_numpy()[arc_ids], arcs.to_node.to_numpy()[arc_ids]
    length = np.hypot(nx[t] - nx[f], ny[t] - ny[f])
    n = np.ceil(length / SAMPLE_M).astype(int) + 1
    owner = np.repeat(np.arange(len(arc_ids)), n)
    start = np.repeat(np.cumsum(n) - n, n)
    frac = (np.arange(len(owner)) - start) / np.maximum(np.repeat(n, n) - 1, 1)
    sx = nx[f][owner] + frac * (nx[t] - nx[f])[owner]
    sy = ny[f][owner] + frac * (ny[t] - ny[f])[owner]
    return np.c_[sx, sy], arc_ids[owner]


def _exact(pi, arc, px, py, arcs, nx, ny):
    """Unique (point, arc) pairs -> exact distance + position along arc."""
    key = np.unique(pi.astype(np.int64) * (len(arcs) + 1) + arc)
    pi, arc = key // (len(arcs) + 1), key % (len(arcs) + 1)
    f, t = arcs.from_node.to_numpy()[arc], arcs.to_node.to_numpy()[arc]
    d, u = point_to_segment(px[pi], py[pi], nx[f], ny[f], nx[t], ny[t])
    return pd.DataFrame({"pi": pi, "cand_arc": arc, "dist_m": d, "proj_u": u})


def generate_candidates(px, py, arcs, nx, ny, radius=RADIUS_M, k=TOP_K):
    """Returns DataFrame(pi, cand_arc, dist_m, proj_u): up to k arcs per point, closest first.
    pi = row position of the point in the cleaned track."""
    arc_ids = crop_arcs(px, py, arcs, nx, ny)
    samples, owner = _arc_samples(arc_ids, arcs, nx, ny)
    tree = cKDTree(samples)
    pts = np.c_[px, py]

    hits = tree.query_ball_point(pts, r=radius + SAMPLE_M / 2)
    lens = np.fromiter((len(h) for h in hits), dtype=int, count=len(hits))
    pi = np.repeat(np.arange(len(pts)), lens)
    cand = _exact(pi, owner[np.concatenate(hits).astype(int)] if lens.sum() else np.array([], int), px, py, arcs, nx, ny)
    cand = cand[cand.dist_m <= radius]

    missing = np.setdiff1d(np.arange(len(pts)), cand.pi.unique())
    if len(missing):
        _, si = tree.query(pts[missing], k=min(40, len(samples)))
        si = np.atleast_2d(si)
        fb = _exact(np.repeat(missing, si.shape[1]), owner[si.ravel()], px, py, arcs, nx, ny)
        fb = fb.sort_values(["pi", "dist_m"]).groupby("pi").head(FALLBACK_K)
        cand = pd.concat([cand, fb], ignore_index=True)

    cand = cand.sort_values(["pi", "dist_m"]).groupby("pi").head(k).reset_index(drop=True)
    return cand
