"""Distance helpers. Trips are worldwide (one city per trip), so degrees are NOT metres and no single UTM zone fits.

We use a local equirectangular projection around each trip (error < 0.1% at city scale),
which turns lon/lat into x/y metres so plain Euclidean geometry works.
"""
import numpy as np

R_EARTH = 6_371_000.0


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def to_xy(lon, lat, lon0, lat0):
    """Project lon/lat (degrees) to local metres around reference point (lon0, lat0)."""
    x = np.radians(np.asarray(lon) - lon0) * R_EARTH * np.cos(np.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * R_EARTH
    return x, y


def bearing_deg(x1, y1, x2, y2):
    """Compass bearing 0..360 (0 = north) of the vector (x1,y1)->(x2,y2) in projected metres."""
    return (np.degrees(np.arctan2(np.asarray(x2) - x1, np.asarray(y2) - y1)) + 360) % 360


def angle_diff(a, b):
    """Smallest absolute difference between two bearings, 0..180."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % 360
    return np.minimum(d, 360 - d)
