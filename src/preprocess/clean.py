"""GPS track cleaning (owner: Samyak). Output is the input of candidate generation.

Rules decided from EDA:
  * drop rows with NaN / out-of-range coordinates
  * sort by time, drop duplicate timestamps (keep first)
  * project lon/lat -> local metres (x, y) around the trip's first point
  * flag time gaps (dt > GAP_S) and impossible jumps (speed > MAX_SPEED) - flagged, NOT dropped,
    so every original GPS point still gets a prediction
"""
import numpy as np

from src.common.geo import bearing_deg, to_xy

GAP_S = 10.0       # a gap longer than this breaks the "1 Hz" assumption
MAX_SPEED = 60.0   # m/s (216 km/h) - anything faster is a GPS glitch


def clean_track(track):
    """track: DataFrame(lon, lat, t) from load_track -> cleaned copy with extra columns.

    Added columns: point_idx (original row number), x, y (metres), dt (s), step_m,
    speed_mps, heading (0..360, from neighbours), is_gap, is_jump.
    """
    df = track.copy()
    df["point_idx"] = np.arange(len(df))
    ok = df[["lon", "lat", "t"]].notna().all(axis=1) & df.lon.between(-180, 180) & df.lat.between(-90, 90)
    df = df[ok].sort_values("t", kind="stable").drop_duplicates("t").reset_index(drop=True)

    lon0, lat0 = df.lon.iloc[0], df.lat.iloc[0]
    df["x"], df["y"] = to_xy(df.lon, df.lat, lon0, lat0)

    dx, dy = np.diff(df.x, prepend=df.x.iloc[0]), np.diff(df.y, prepend=df.y.iloc[0])
    df["dt"] = df.t.diff().fillna(0.0)
    df["step_m"] = np.hypot(dx, dy)
    df["speed_mps"] = np.where(df.dt > 0, df.step_m / df.dt.replace(0, np.nan), 0.0)
    df["speed_mps"] = df.speed_mps.fillna(0.0)

    # heading from previous point to next point (centred difference = less noisy than one step)
    xp, yp = df.x.shift(1).fillna(df.x), df.y.shift(1).fillna(df.y)
    xn, yn = df.x.shift(-1).fillna(df.x), df.y.shift(-1).fillna(df.y)
    df["heading"] = bearing_deg(xp, yp, xn, yn)

    df["is_gap"] = df.dt > GAP_S
    df["is_jump"] = df.speed_mps > MAX_SPEED
    df.attrs["origin"] = (lon0, lat0)  # needed to project nodes into the same x/y frame
    return df
