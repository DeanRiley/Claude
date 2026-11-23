"""
source.py — the common "spectral source" dict every reader returns.

A source is a plain dict:

    kind        "spectra" (1-D / 2-D tables, libraries) or "image" (cube)
    wl          (B,) band centres, nm
    fwhm        (B,) nm or None
    data        spectra: (N, B) array;  image: (rows, cols, B) array or None
    read_block  image only: f(r0, r1) -> (r1-r0, cols, B) float array
    shape       image only: (rows, cols, B)
    meta        spectra only: pandas DataFrame (N rows) of IDs / attributes
    labels      spectra only: list of N row labels
    scale       multiplier to reflectance (None -> auto-detect)
    crs, transform   georeferencing (rasterio CRS / Affine) or None
    sensor, path     free text
"""
from __future__ import annotations

import numpy as np


def new_source(kind, wl, **kw):
    src = {"kind": kind, "wl": np.asarray(wl, float).ravel(), "fwhm": None,
           "data": None, "read_block": None, "shape": None, "meta": None,
           "labels": None, "scale": None, "crs": None, "transform": None,
           "sensor": "", "path": ""}
    src.update(kw)
    if kind == "image" and src["read_block"] is None and src["data"] is not None:
        arr = src["data"]
        src["shape"] = arr.shape
        src["read_block"] = lambda r0, r1, _a=arr: np.asarray(_a[r0:r1], float)
    return src


def to_nm(wl, units=None):
    """Convert wavelengths to nm. units: 'um','micrometers','nm', None=auto."""
    wl = np.asarray(wl, float)
    u = (units or "").lower()
    if u.startswith(("micro", "um", "µm", "μm")):
        return wl * 1000.0
    if u.startswith(("nano", "nm")):
        return wl
    if u in ("wavenumber", "cm-1", "1/cm"):
        return 1e7 / wl
    finite = wl[np.isfinite(wl) & (wl > 0)]
    return wl * 1000.0 if finite.size and np.nanmax(finite) < 30 else wl


def auto_scale(sample, warn=True):
    """Guess the factor that converts stored values to 0-1 reflectance."""
    v = np.asarray(sample, float)
    v = v[np.isfinite(v) & (v > -1e30)]
    if v.size == 0:
        return 1.0
    p99 = np.percentile(v, 99)
    if p99 > 200:          # 0-10000 integers (AVIRIS, EnMAP, PRISMA DN, ...)
        f = 1e-4
    elif p99 > 2.0:        # percent reflectance (lab exports)
        f = 1e-2
    else:
        f = 1.0
    if warn and f != 1.0:
        print(f"[crism-indices] auto scale: 99th percentile {p99:.3g} -> x{f:g}")
    return f
