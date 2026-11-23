"""
browse.py — CRISM browse products (RGB composites of summary parameters)
from Viviano-Beck et al. (2014), Table 3, plus stretching and PNG output.

Products marked (*) were not checked against the paper's Table 3 in this
build; verify before publishing.
"""
from __future__ import annotations

import numpy as np

BROWSE = {
    # VNIR
    "TRU": ("R600", "R530", "R440"),
    "FEM": ("BD530_2", "SH600_2", "BDI1000VIS"),
    "FM2": ("BD530_2", "BD920_2", "BDI1000VIS"),
    # IR
    "FAL": ("R2529", "R1506", "R1080"),
    "MAF": ("OLINDEX3", "LCPINDEX2", "HCPINDEX2"),
    "HYD": ("SINDEX2", "BD2100_2", "BD1900_2"),
    "PHY": ("D2300", "D2200", "BD1900r2"),
    "PFM": ("BD2355", "D2300", "BD2290"),
    "PAL": ("BD2210_2", "BD2190", "BD2165"),
    "HYS": ("MIN2250", "BD2250", "BD1900r2"),
    "ICE": ("BD1900_2", "BD1500_2", "BD1435"),
    "IC2": ("R3920", "BD1500_2", "BD1435"),
    "CHL": ("ISLOPE1", "BD3000", "IRR2"),       # (*)
    "CAR": ("D2300", "BD2500_2", "BD1900_2"),    # (*)
    "CR2": ("MIN2295_2480", "MIN2345_2537", "CINDEX2"),
}

REFLECTANCE_PRODUCTS = {"TRU", "FAL"}


def stretch_limits(values, product, pct=(1, 99)):
    """Lower/upper stretch for one channel. Parameter products: lower 0,
    upper = pct[1] percentile of positive values. Reflectance products:
    pct[0]..pct[1] percentiles."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    if product in REFLECTANCE_PRODUCTS:
        lo, hi = np.percentile(v, pct)
    else:
        pos = v[v > 0]
        lo, hi = 0.0, (np.percentile(pos, pct[1]) if pos.size else 1.0)
    return float(lo), float(hi if hi > lo else lo + 1e-6)


def browse_rgb(params, product, limits=None, pct=(1, 99)):
    """params: dict name -> array (any shape). Returns float RGB in 0-1,
    shape (..., 3), NaN -> 0, plus the limits used."""
    names = BROWSE[product]
    chans, used = [], []
    for i, n in enumerate(names):
        v = params.get(n)
        if v is None:
            raise KeyError(f"{product} needs parameter {n}")
        lo, hi = limits[i] if limits else stretch_limits(v, product, pct)
        used.append((lo, hi))
        with np.errstate(all="ignore"):
            chans.append(np.clip((np.asarray(v, float) - lo) / (hi - lo), 0, 1))
    return np.nan_to_num(np.stack(chans, -1), nan=0.0), used


def save_png(rgb, path, transform=None, crs=None):
    """Write an RGB PNG; with a transform also write a world file (.pgw)
    and a .prj (geo-PNG)."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    plt.imsave(path, (np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    if transform is not None:
        t = transform
        # centre of upper-left pixel
        Path(path).with_suffix(".pgw").write_text(
            "\n".join(str(x) for x in (t.a, t.d, t.b, t.e,
                                       t.c + t.a / 2 + t.b / 2, t.f + t.d / 2 + t.e / 2)) + "\n")
        if crs is not None:
            try:
                Path(path).with_suffix(".prj").write_text(crs.to_wkt())
            except Exception:
                pass
    return path
