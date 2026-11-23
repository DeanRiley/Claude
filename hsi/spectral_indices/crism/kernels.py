"""
kernels.py — core reflectance-extraction / continuum / band-depth operations
for CRISM-style summary parameters (Viviano-Beck et al., 2014), generalized
so they work on any sensor's sampling.

All functions are plain functions operating on a "context" dict built by
make_context(). The spectral axis is always the LAST axis of ctx["d"].

Kernel modes
------------
"nm"    (default) A CRISM kernel of k bands is converted to a wavelength
        window: half-width = (k//2)*6.55 nm + 6.55/2. All input bands inside
        that window (centred on the band nearest the target λ) are median-ed.
        This gives CRISM-equivalent smoothing on lab (1 nm), airborne
        (5-10 nm) and spaceborne (7-12 nm) data alike; on coarse sensors it
        collapses to the nearest band.
"bands" Use k input bands literally (exact CAT behaviour on CRISM data).
"""
from __future__ import annotations

import warnings
import numpy as np

CRISM_STEP_NM = 6.55
INVALID_VALUES = (65535.0, -9999.0)


def make_context(data, wl_nm, kernel_mode="nm", tol_nm=None, anchor_tol_nm=40.0):
    """Build the processing context.

    data   : array (..., B) reflectance (any leading shape)
    wl_nm  : (B,) band centres in nm (need not be sorted; NaN allowed)
    tol_nm : max distance from a target λ to the nearest band for that λ to
             count as covered. Default max(10 nm, 0.75 * median spacing).
    anchor_tol_nm : CONTINUUM ANCHORS (never band centres) may be taken from
             the nearest valid band up to this distance away. This covers
             (a) anchors beyond the sensor edge (2530 nm on 2.5 µm VSWIR
             sensors: D2300, HCPINDEX2, ISLOPE1) and (b) anchors inside
             bad-band windows (1815 nm in the 1.8-1.95 µm H2O window:
             D2200, D2300, BD1900r2). Substitutions are logged in
             ctx["subs"]. Set 0 for strict CRISM behaviour.
    """
    d = np.asarray(data, dtype=np.float64)
    wl = np.asarray(wl_nm, dtype=np.float64).ravel()
    if d.shape[-1] != wl.size:
        raise ValueError(f"spectral axis ({d.shape[-1]}) != n wavelengths ({wl.size})")
    for v in INVALID_VALUES:
        d = np.where(d == v, np.nan, d)
    order = np.argsort(np.where(np.isfinite(wl), wl, np.inf))
    if np.any(order != np.arange(wl.size)):
        d, wl = d[..., order], wl[order]
    good = np.isfinite(wl)
    d, wl = d[..., good], wl[good]
    # bands that are NaN everywhere (bad bands / atmospheric windows) are dropped
    allnan = np.all(~np.isfinite(d.reshape(-1, wl.size)), axis=0)
    if allnan.any() and not allnan.all():
        d, wl = d[..., ~allnan], wl[~allnan]
    spacing = np.median(np.diff(wl)) if wl.size > 1 else CRISM_STEP_NM
    if tol_nm is None:
        tol_nm = max(10.0, 0.75 * spacing)
    return {"d": d, "wl": wl, "mode": kernel_mode, "tol": tol_nm, "anchor_tol": anchor_tol_nm,
            "subs": set(),
            "spacing": spacing, "cache": {}, "shape": d.shape[:-1]}


def nearest_index(ctx, lam):
    return int(np.argmin(np.abs(ctx["wl"] - lam)))


def W(ctx, lam):
    """Centre wavelength of the band nearest lam."""
    return ctx["wl"][nearest_index(ctx, lam)]


def covered(ctx, *lams, anchor=False):
    """True if every lam has a band within tol (or anchor_tol for anchors)."""
    wl = ctx["wl"]
    if wl.size == 0:
        return False
    lim = max(ctx["tol"], ctx["anchor_tol"]) if anchor else ctx["tol"]
    return all(abs(W(ctx, l) - l) <= lim for l in lams)


def nan_array(ctx):
    return np.full(ctx["shape"], np.nan)


def window_indices(ctx, lam, k):
    i = nearest_index(ctx, lam)
    k = int(k) | 1
    if ctx["mode"] == "bands":
        h = k // 2
        return max(0, i - h), min(ctx["wl"].size, i + h + 1)
    hw = (k // 2) * CRISM_STEP_NM + CRISM_STEP_NM / 2.0
    c = ctx["wl"][i]
    inside = np.nonzero(np.abs(ctx["wl"] - c) <= hw)[0]
    if inside.size == 0:
        return i, i + 1
    return int(inside[0]), int(inside[-1]) + 1


def R(ctx, lam, k=5, anchor=False):
    """Median-kernel reflectance at lam (anchor=True allows substitution)."""
    key = ("R", lam, k, anchor)
    c = ctx["cache"]
    if key in c:
        return c[key]
    if anchor and not covered(ctx, lam) and covered(ctx, lam, anchor=True):
        ctx["subs"].add(lam)
    if not covered(ctx, lam, anchor=anchor):
        out = nan_array(ctx)
    else:
        lo, hi = window_indices(ctx, lam, k)
        if hi - lo == 1:
            out = ctx["d"][..., lo]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out = np.nanmedian(ctx["d"][..., lo:hi], axis=-1)
    c[key] = out
    return out


def RC(ctx, lam, lo, hi, klo=5, khi=5):
    """Linear continuum between anchors lo and hi, evaluated at W(lam)."""
    wl_, wc, wh = W(ctx, lo), W(ctx, lam), W(ctx, hi)
    if wh == wl_:
        return nan_array(ctx)
    a = (wc - wl_) / (wh - wl_)
    return (1.0 - a) * R(ctx, lo, klo, anchor=True) + a * R(ctx, hi, khi, anchor=True)


def RB(ctx, lam, lo, hi, kc=5, klo=5, khi=5):
    """Band depth 1 - R/RC."""
    with np.errstate(all="ignore"):
        return 1.0 - R(ctx, lam, kc) / RC(ctx, lam, lo, hi, klo, khi)


def ratio_to_continuum(ctx, lam, lo, hi, kc=1, klo=5, khi=5):
    with np.errstate(all="ignore"):
        return R(ctx, lam, kc) / RC(ctx, lam, lo, hi, klo, khi)


def bands_in(ctx, lo, hi, lo_inclusive=True):
    wl = ctx["wl"]
    m = (wl >= lo) if lo_inclusive else (wl > lo)
    return np.nonzero(m & (wl <= hi))[0]
