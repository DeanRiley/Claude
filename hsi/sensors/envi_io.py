"""
envi_io.py — ENVI cubes (.hdr + .img/.dat/.bsq/.bil/.bip/none).

Covers most airborne products: AVIRIS-Classic / AVIRIS-NG / AVIRIS-3 L2
reflectance, HySpex (Mjolnir, ODIN), AISA (Eagle/Hawk/Dual/Fenix),
HyMap, CASI/SASI, SpecTIR, Headwall, CRISM cubes exported by CAT, and
lab imaging spectrometers (SisuROCK, Specim) — plus ENVI spectral
libraries (file type = ENVI Spectral Library -> spectra source).

Reads lazily via spectral's memmap; georeferencing via rasterio when
installed (GDAL understands ENVI 'map info' / 'coordinate system string').
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .source import new_source, to_nm


def find_header(path):
    p = Path(path)
    if p.suffix.lower() == ".hdr":
        return p
    for cand in (p.with_suffix(".hdr"), Path(str(p) + ".hdr")):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"no ENVI header for {p}")


def find_data(hdr):
    hdr = Path(hdr)
    for ext in ("", ".img", ".dat", ".bsq", ".bil", ".bip", ".raw", ".IMG", ".DAT"):
        cand = hdr.with_suffix(ext) if ext else hdr.with_suffix("")
        if cand.exists() and cand != hdr:
            return cand
    raise FileNotFoundError(f"no ENVI data file next to {hdr}")


def read_envi(path, wavelengths=None, wl_units=None):
    import spectral.io.envi as envi
    hdr = find_header(path)
    md = envi.read_envi_header(str(hdr))
    if "spectral library" in md.get("file type", "").lower():
        from .tabular_io import read_envi_sli
        return read_envi_sli(hdr)
    img = envi.open(str(hdr), str(find_data(hdr)))
    rows, cols, nb = img.shape

    if wavelengths is not None:
        wl = to_nm(np.asarray(wavelengths, float), wl_units)
    elif "wavelength" in md:
        wl = to_nm(np.asarray(md["wavelength"], float), wl_units or md.get("wavelength units"))
    else:
        raise ValueError(f"{hdr} has no 'wavelength' field; pass wavelengths=")
    fwhm = (to_nm(np.asarray(md["fwhm"], float), md.get("wavelength units"))
            if "fwhm" in md else None)

    bbl = np.asarray(md.get("bbl", np.ones(nb)), float).astype(bool)
    ignore = md.get("data ignore value")
    ignore = float(ignore) if ignore is not None else None
    # spectral applies 'reflectance scale factor' itself when reading
    scale = 1.0 if md.get("reflectance scale factor") not in (None, "", "0") else None

    def read_block(r0, r1, _img=img):
        blk = np.asarray(_img.read_subregion((r0, r1), (0, cols)), float)
        if ignore is not None:
            blk[blk == ignore] = np.nan
        blk[..., ~bbl] = np.nan
        return blk

    crs = transform = None
    try:
        import rasterio
        with rasterio.open(str(find_data(hdr))) as ds:
            crs, transform = ds.crs, ds.transform
    except Exception:
        pass
    return new_source("image", wl, fwhm=fwhm, read_block=read_block,
                      shape=(rows, cols, nb), scale=scale, crs=crs, transform=transform,
                      sensor=md.get("sensor type", "ENVI"), path=str(hdr), meta_header=md,
                      good_bands=bbl)
