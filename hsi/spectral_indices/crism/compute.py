"""
compute.py — compute CRISM summary parameters on anything spectral.

    from hsi.spectral_indices.crism.compute import compute

    compute(spectrum_array, wavelengths=wl)          -> pandas Series
    compute(df)                                      -> DataFrame (meta + params)
    compute("lab_spectra.xlsx")                      -> DataFrame (all sheets)
    compute("ang20230912t200508_rfl.hdr",
            out="ang_crism.tif")                     -> streams GeoTIFF to disk
    compute("EMIT_L2A_RFL_....nc", ortho=True,
            out="emit_crism.tif", browse=["PHY","PAL","MAF"])
    compute(cube, wavelengths=wl)                    -> dict of 2-D arrays

Parameter selection (names=)
    "auto"         every parameter the sensor's wavelengths cover (default)
    "terrestrial"  VSWIR subset avoiding atmospheric windows
    "fig9"         the 26 IR parameters of Viviano-Beck Fig. 9
    "all"          everything (uncovered ones come back NaN)
    list           explicit names
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import kernels as K
from .parameters import PARAMETERS, TERRESTRIAL, FIG9_COLUMNS
from .browse import BROWSE, browse_rgb, save_png

CRISM_IR_WL = np.arange(1001.35, 3936.4, 6.55)
CRISM_VNIR_WL = np.arange(362.0, 1053.0, 6.55)


# ---------------------------------------------------------------------------
# coverage / names
# ---------------------------------------------------------------------------
def coverage_table(wl_nm, names=None, tol_nm=None, anchor_tol_nm=40.0, good=None):
    """Which parameters a band set supports, found by evaluating every
    parameter on a flat spectrum (so it is always consistent with compute).
    good: optional boolean mask of usable bands (bbl / good_wavelengths).
    Columns: covered, substituted_anchors_nm (anchors taken from a
    neighbouring band, see kernels.make_context)."""
    wl = np.asarray(wl_nm, float)
    flat = np.ones((1, wl.size))
    if good is not None:
        flat[:, ~np.asarray(good, bool)] = np.nan
    rows = []
    for n in names or PARAMETERS:
        ctx = K.make_context(flat, wl, tol_nm=tol_nm, anchor_tol_nm=anchor_tol_nm)
        with np.errstate(all="ignore"):
            v = np.asarray(PARAMETERS[n]["func"](ctx), float)
        ok = bool(np.isfinite(v).all())
        rows.append({"parameter": n, "group": PARAMETERS[n]["group"],
                     "target": PARAMETERS[n]["target"], "covered": ok,
                     "substituted_anchors_nm": ",".join(f"{s:g}" for s in sorted(ctx["subs"]))
                     if ok else ""})
    return pd.DataFrame(rows)


def resolve_names(names, wl_nm, extra=(), anchor_tol_nm=40.0, good=None):
    if isinstance(names, str) and names not in ("auto", "all", "terrestrial", "fig9"):
        names = [names]
    if names is None or names == "auto":
        cov = coverage_table(wl_nm, anchor_tol_nm=anchor_tol_nm, good=good)
        out = cov.loc[cov.covered, "parameter"].tolist()
    elif names == "all":
        out = list(PARAMETERS)
    elif names == "terrestrial":
        out = list(TERRESTRIAL)
    elif names == "fig9":
        out = list(FIG9_COLUMNS)
    else:
        out = list(names)
    for e in extra:
        if e not in out:
            out.append(e)
    bad = [n for n in out if n not in PARAMETERS]
    if bad:
        raise KeyError(f"unknown parameters: {bad}")
    return out


# ---------------------------------------------------------------------------
# resampling (e.g. 1 nm lab spectra -> CRISM bands)
# ---------------------------------------------------------------------------
def gaussian_resample(data, wl_src, wl_dst, fwhm_dst=11.0):
    """Gaussian-convolve (..., B_src) onto wl_dst. NaNs ignored."""
    d = np.asarray(data, float)
    ws = np.asarray(wl_src, float)
    wd = np.asarray(wl_dst, float)
    fw = np.broadcast_to(np.asarray(fwhm_dst, float), wd.shape)
    sig = fw / 2.3548
    Wt = np.exp(-0.5 * ((ws[None, :] - wd[:, None]) / sig[:, None]) ** 2)   # (Bd, Bs)
    Wt[(wd < ws.min() - sig) | (wd > ws.max() + sig)] = 0
    good = np.isfinite(d)
    num = np.where(good, d, 0.0) @ Wt.T
    den = good.astype(float) @ Wt.T
    with np.errstate(all="ignore"):
        out = num / den
    out[den < 1e-3] = np.nan
    return out


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------
def compute_array(data, wl_nm, names="auto", kernel_mode="nm", tol_nm=None,
                  anchor_tol_nm=40.0):
    """data (..., B) reflectance -> {name: array(...)}."""
    names = resolve_names(names, wl_nm, anchor_tol_nm=anchor_tol_nm)
    ctx = K.make_context(data, wl_nm, kernel_mode, tol_nm, anchor_tol_nm)
    return {n: np.asarray(PARAMETERS[n]["func"](ctx), float) for n in names}


def _prep(src, scale, resample_to, fwhm):
    wl = src["wl"]
    if resample_to is None:
        return wl, None
    tgt = {"crism": CRISM_IR_WL, "crism_ir": CRISM_IR_WL,
           "crism_vnir_ir": np.r_[CRISM_VNIR_WL[CRISM_VNIR_WL < 1001], CRISM_IR_WL]
           }.get(resample_to, resample_to) if isinstance(resample_to, str) else resample_to
    tgt = np.asarray(tgt, float)
    return tgt, (lambda d: gaussian_resample(d, wl, tgt, fwhm))


def _scale_factor(src, scale, sample):
    if scale is not None:
        return float(scale)
    if src.get("scale") is not None:
        return float(src["scale"])
    from hsi.sensors.source import auto_scale
    return auto_scale(sample)


def compute(obj, names="auto", wavelengths=None, kernel_mode="nm", scale=None,
            resample_to=None, fwhm=11.0, tol_nm=None, anchor_tol_nm=40.0, out=None, out_format=None,
            browse=None, browse_dir=None, block_lines=128, verbose=True, **reader_kw):
    """Compute CRISM parameters for any spectral input. See module docstring.

    scale       multiplier to 0-1 reflectance (None: from file metadata, else auto)
    resample_to None | "crism" | "crism_vnir_ir" | array of nm — Gaussian
                resample first (fwhm nm), e.g. to emulate CRISM from lab data
    kernel_mode "nm" (CRISM-equivalent windows, default) or "bands"
    anchor_tol_nm  continuum anchors may come from the nearest valid band up to
                this far away (sensor edge at 2.5 µm, H2O bad-band windows).
                Default 40 nm; 0 = strict CRISM. See coverage_table().
    out         image only: output path (.tif, .img/.hdr ENVI, .nc, .npz).
                Without out, image results are returned in memory.
    browse      list of browse products (e.g. ["PHY","MAF"]) -> PNGs (+ .pgw)
    """
    from hsi.sensors.open_any import open_spectral
    src = open_spectral(obj, wavelengths=wavelengths, **reader_kw)
    wl_use, resampler = _prep(src, scale, resample_to, fwhm)
    extra = sorted({c for b in (browse or []) for c in BROWSE[b]})
    good = None if resampler else src.get("good_bands")
    names_ = resolve_names(names, wl_use, extra, anchor_tol_nm, good)

    if src["kind"] == "spectra":
        d = np.asarray(src["data"], float)
        d = d * _scale_factor(src, scale, d)
        if resampler:
            d = resampler(d)
        res = compute_array(d, wl_use, names_, kernel_mode, tol_nm, anchor_tol_nm)
        df = pd.DataFrame(res)
        meta = src.get("meta")
        if meta is None:
            meta = pd.DataFrame({"label": src["labels"]})
        df = pd.concat([meta.reset_index(drop=True), df], axis=1)
        if src.get("squeeze"):
            return df.iloc[0][names_]
        return df

    # ---------------- image ----------------
    rows, cols, nb = src["shape"]
    first = src["read_block"](0, min(rows, 16))
    f = _scale_factor(src, scale, first)
    writer = None
    if out is not None:
        from .outputs import open_writer
        writer = open_writer(out, out_format, rows, cols, names_, src.get("crs"),
                             src.get("transform"), wl_use=wl_use)
        result = None
    else:
        result = {n: np.full((rows, cols), np.nan, np.float32) for n in names_}
    t0 = time.time()
    for r0 in range(0, rows, block_lines):
        r1 = min(rows, r0 + block_lines)
        blk = src["read_block"](r0, r1) * f
        if resampler:
            blk = resampler(blk)
        res = compute_array(blk, wl_use, names_, kernel_mode, tol_nm, anchor_tol_nm)
        if writer:
            writer["write"](r0, r1, res)
        else:
            for n in names_:
                result[n][r0:r1] = res[n]
        if verbose:
            print(f"\r[crism-indices] lines {r1}/{rows}  {time.time() - t0:5.1f}s", end="")
    if verbose:
        print()
    meta = {"names": names_, "crs": src.get("crs"), "transform": src.get("transform"),
            "wl": wl_use, "sensor": src.get("sensor"), "source": src.get("path")}
    if writer:
        writer["close"]()
        meta["path"] = str(out)
        if browse:
            from .outputs import read_params
            result = read_params(out, extra)
    if browse:
        from pathlib import Path
        bdir = Path(browse_dir or (Path(out).parent if out else "."))
        bdir.mkdir(parents=True, exist_ok=True)
        stem = Path(out).stem if out else "crism"
        meta["browse"] = {}
        for b in browse:
            rgb, lim = browse_rgb(result, b)
            pth = bdir / f"{stem}_{b}.png"
            save_png(rgb, pth, src.get("transform"), src.get("crs"))
            meta["browse"][b] = {"png": str(pth), "limits": lim}
    if writer:
        return meta
    meta["params"] = result
    return meta
