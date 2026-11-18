"""Shared loaders for raw PathSnap trips. Every member uses these, never re-parse files by hand.

A trip folder looks like  <root>/<trip_id>/<trip_id>.{track,nodes,arcs,route}
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PARTITIONS = {"train": RAW / "train", "student_test": RAW / "student_test", "eval": ROOT / "data" / "eval"}


def list_trips(partition):
    """Sorted trip ids (folder names) in a partition: 'train', 'student_test' or 'eval'."""
    folder = PARTITIONS[partition]
    return sorted(p.name for p in folder.iterdir() if p.is_dir())


def trip_dir(partition, trip_id):
    return PARTITIONS[partition] / trip_id


def load_track(d):
    d = Path(d)
    return pd.read_csv(d / f"{d.name}.track", sep=r"\s+", header=None, names=["lon", "lat", "t"])


def load_nodes(d):
    d = Path(d)
    df = pd.read_csv(d / f"{d.name}.nodes", sep=r"\s+", header=None, names=["lon", "lat"])
    df.index.name = "node_id"  # 0-based row index is the node id
    return df


def load_arcs(d):
    d = Path(d)
    df = pd.read_csv(d / f"{d.name}.arcs", sep=r"\s+", header=None, names=["from_node", "to_node"])
    df.index.name = "arc_id"  # 0-based row index is the arc id; arcs are DIRECTED
    return df


def load_route(d):
    """Ordered list of ground-truth arc ids. Returns None if the trip has no .route file."""
    d = Path(d)
    f = d / f"{d.name}.route"
    if not f.exists():
        return None
    return pd.read_csv(f, header=None).iloc[:, 0].astype(int).tolist()


def load_trip(d):
    d = Path(d)
    return {
        "trip_id": d.name,
        "track": load_track(d),
        "nodes": load_nodes(d),
        "arcs": load_arcs(d),
        "route": load_route(d),
    }
