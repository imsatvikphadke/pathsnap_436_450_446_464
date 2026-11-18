"""Folium map builder (owner: Samyak). Used by the report maps AND by the API for the UI."""
import folium
import numpy as np

# one fixed colour per layer name so every map looks the same
COLOURS = {
    "Reference route": "#2ca02c",   # green
    "Classifier only": "#ff7f0e",   # orange (dashed)
    "HMM / Viterbi": "#d62728",     # red
    "Predicted route": "#d62728",   # red (UI map)
}
MAX_GPS_MARKERS = 3000


def _segments(arc_ids, nodes, arcs):
    """arc ids -> list of [[lat, lon], [lat, lon]] pieces; -1 / out-of-range ids are skipped."""
    ids = np.asarray([a for a in arc_ids if a is not None], dtype=np.int64)
    ids = ids[(ids >= 0) & (ids < len(arcs))]
    f = arcs.from_node.to_numpy()[ids]
    t = arcs.to_node.to_numpy()[ids]
    lat, lon = nodes.lat.to_numpy(), nodes.lon.to_numpy()
    return [[[lat[a], lon[a]], [lat[b], lon[b]]] for a, b in zip(f, t)]


def _dedupe(seq):
    """Drop consecutive repeats (one arc is usually matched by many GPS points)."""
    return [a for i, a in enumerate(seq) if i == 0 or a != seq[i - 1]]


def make_route_map(track, nodes, arcs, routes=None, show_network=True, buffer_deg=0.002, max_net_arcs=15000):
    """track/nodes/arcs = DataFrames from src.common.io; routes = {"layer name": [arc ids...]} -> folium.Map."""
    routes = routes or {}
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)

    if show_network:
        lo0, lo1 = track.lon.min() - buffer_deg, track.lon.max() + buffer_deg
        la0, la1 = track.lat.min() - buffer_deg, track.lat.max() + buffer_deg
        inside = (nodes.lon.between(lo0, lo1) & nodes.lat.between(la0, la1)).to_numpy()
        keep = np.flatnonzero(inside[arcs.from_node.to_numpy()] & inside[arcs.to_node.to_numpy()])[:max_net_arcs]
        net = folium.FeatureGroup(name=f"Road network ({len(keep)} arcs)")
        if len(keep):
            folium.PolyLine(_segments(keep, nodes, arcs), color="#666666", weight=1, opacity=0.5).add_to(net)
        net.add_to(m)

    step = int(np.ceil(len(track) / MAX_GPS_MARKERS)) if len(track) > MAX_GPS_MARKERS else 1
    gps_name = f"GPS observations ({len(track)})" + (f" - every {step}th shown" if step > 1 else "")
    gps = folium.FeatureGroup(name=gps_name)
    for lat, lon, t in track[["lat", "lon", "t"]].iloc[::step].itertuples(index=False):
        folium.CircleMarker([lat, lon], radius=2, color="#1f77b4", fill=True, fill_opacity=0.8, weight=0,
                            tooltip=f"t={t:.0f}s").add_to(gps)
    gps.add_to(m)

    for name, arc_ids in routes.items():
        fg = folium.FeatureGroup(name=name)
        seg = _segments(_dedupe(list(arc_ids)), nodes, arcs)
        if seg:
            folium.PolyLine(seg, color=COLOURS.get(name, "#9467bd"), weight=7 if name == "Reference route" else 4,
                            opacity=0.75, dash_array="8" if name == "Classifier only" else None).add_to(fg)
        fg.add_to(m)

    folium.Marker(track[["lat", "lon"]].iloc[0].tolist(), tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(track[["lat", "lon"]].iloc[-1].tolist(), tooltip="End", icon=folium.Icon(color="red")).add_to(m)

    items = [(gps_name, "#1f77b4"), ("Road network", "#666666")] + [(n, COLOURS.get(n, "#9467bd")) for n in routes]
    rows = "".join(f'<div><span style="background:{c};width:18px;height:4px;display:inline-block;margin-right:6px"></span>{n}</div>'
                   for n, c in items)
    legend = (f'<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px 12px;'
              f'border:1px solid #999;border-radius:6px;font-size:13px">{rows}</div>')
    m.get_root().html.add_child(folium.Element(legend))

    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[track.lat.min(), track.lon.min()], [track.lat.max(), track.lon.max()]])
    return m
