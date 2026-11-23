"""
spaceborne_io.py — spaceborne imaging spectrometer readers.

EMIT   L2A RFL netCDF  (EMIT_L2A_RFL_*.nc)   swath or GLT-orthorectified
PRISMA L2B/L2C/L2D HE5 (PRS_L2?_STD_*.he5)    VNIR + SWIR cubes merged
EnMAP  L2A             (*-SPECTRAL_IMAGE.TIF + *-METADATA.XML, or the folder)
DESIS / others with GeoTIFF + XML  -> geotiff_io.read_geotiff(xml=...)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .source import new_source, to_nm


# ---------------------------------------------------------------------------
# EMIT
# ---------------------------------------------------------------------------
def read_emit(path, ortho=False, mask_bad=True):
    """EMIT L2A reflectance. ortho=True applies the GLT (output on the
    geographic grid in the file's 'geotransform'); otherwise swath geometry."""
    import netCDF4
    ds = netCDF4.Dataset(str(path))
    ds.set_auto_mask(False)
    var = ds["reflectance"]
    sbp = ds.groups["sensor_band_parameters"]
    wl = to_nm(np.asarray(sbp["wavelengths"][:], float))
    fwhm = to_nm(np.asarray(sbp["fwhm"][:], float)) if "fwhm" in sbp.variables else None
    good = (np.asarray(sbp["good_wavelengths"][:]).astype(bool)
            if (mask_bad and "good_wavelengths" in sbp.variables) else np.ones(wl.size, bool))
    fill = getattr(var, "_FillValue", -9999.0)
    nl, ns, nb = var.shape

    def clean(a):
        a = np.asarray(a, float)
        a[a == fill] = np.nan
        a[a == -9999] = np.nan
        a[..., ~good] = np.nan
        return a

    crs = transform = None
    if not ortho:
        read_block = lambda r0, r1: clean(var[r0:r1, :, :])
        shape = (nl, ns, nb)
    else:
        loc = ds.groups["location"]
        gx = np.asarray(loc["glt_x"][:], np.int64)
        gy = np.asarray(loc["glt_y"][:], np.int64)
        gt = list(ds.getncattr("geotransform"))
        from rasterio.transform import Affine
        from rasterio.crs import CRS
        transform = Affine.from_gdal(*gt)
        crs = CRS.from_wkt(ds.getncattr("spatial_ref")) if "spatial_ref" in ds.ncattrs() else CRS.from_epsg(4326)
        shape = (gx.shape[0], gx.shape[1], nb)

        def read_block(r0, r1):
            bx, by = np.abs(gx[r0:r1]), np.abs(gy[r0:r1])   # 1-based, 0 = none
            valid = (bx > 0) & (by > 0)
            out = np.full((r1 - r0, gx.shape[1], nb), np.nan)
            if valid.any():
                y0, y1 = by[valid].min() - 1, by[valid].max()
                slab = clean(var[y0:y1, :, :])
                out[valid] = slab[by[valid] - 1 - y0, bx[valid] - 1]
            return out

    return new_source("image", wl, fwhm=fwhm, read_block=read_block, shape=shape,
                      scale=1.0, crs=crs, transform=transform, sensor="EMIT", path=str(path),
                      good_bands=good)


# ---------------------------------------------------------------------------
# PRISMA
# ---------------------------------------------------------------------------
def read_prisma(path):
    import h5py
    f = h5py.File(str(path), "r")
    level = next(k for k in f["HDFEOS/SWATHS"].keys() if k.startswith("PRS_L2"))
    dfld = f[f"HDFEOS/SWATHS/{level}/Data Fields"]
    vn, sw = dfld["VNIR_Cube"], dfld["SWIR_Cube"]     # (lines, bands, samples)
    a = f.attrs
    wv, ws = np.asarray(a["List_Cw_Vnir"], float), np.asarray(a["List_Cw_Swir"], float)
    fv = np.asarray(a.get("List_Fwhm_Vnir", np.zeros_like(wv)), float)
    fs = np.asarray(a.get("List_Fwhm_Swir", np.zeros_like(ws)), float)
    smin_v, smax_v = float(a.get("L2ScaleVnirMin", 0)), float(a.get("L2ScaleVnirMax", 1))
    smin_s, smax_s = float(a.get("L2ScaleSwirMin", 0)), float(a.get("L2ScaleSwirMax", 1))
    dmax = 65535.0
    # drop flagged (zero) bands and VNIR bands overlapping SWIR
    swmin = ws[ws > 0].min()
    kv = np.nonzero((wv > 0) & (wv < swmin))[0]
    ks = np.nonzero(ws > 0)[0]
    wl = np.r_[wv[kv], ws[ks]]
    fwhm = np.r_[fv[kv], fs[ks]]
    nl, _, ns = vn.shape

    def read_block(r0, r1):
        v = np.asarray(vn[r0:r1, :, :], float)[:, kv, :]
        s = np.asarray(sw[r0:r1, :, :], float)[:, ks, :]
        v = smin_v + v * (smax_v - smin_v) / dmax
        s = smin_s + s * (smax_s - smin_s) / dmax
        blk = np.concatenate([v, s], axis=1)                 # (rows, B, samples)
        return np.moveaxis(blk, 1, -1)

    crs = transform = None
    if level.startswith("PRS_L2D") and "Epsg_Code" in a:
        from rasterio.transform import from_origin
        from rasterio.crs import CRS
        crs = CRS.from_epsg(int(a["Epsg_Code"]))
        transform = from_origin(float(a["Product_ULcorner_easting"]),
                                float(a["Product_ULcorner_northing"]), 30.0, 30.0)
    return new_source("image", wl, fwhm=fwhm, read_block=read_block, shape=(nl, ns, wl.size),
                      scale=1.0, crs=crs, transform=transform, sensor="PRISMA " + level,
                      path=str(path))


# ---------------------------------------------------------------------------
# EnMAP
# ---------------------------------------------------------------------------
def read_enmap(path):
    """path: product folder, the SPECTRAL_IMAGE GeoTIFF, or the METADATA.XML."""
    from .geotiff_io import read_geotiff, wavelengths_from_xml
    p = Path(path)
    folder = p if p.is_dir() else p.parent
    tif = (p if p.suffix.lower() in (".tif", ".tiff") else
           next(iter(sorted(folder.glob("*SPECTRAL_IMAGE*.TIF")) + sorted(folder.glob("*SPECTRAL_IMAGE*.tif")))))
    xml = next(iter(sorted(folder.glob("*METADATA*.XML")) + sorted(folder.glob("*METADATA*.xml"))))
    wl, fwhm, gain, off = wavelengths_from_xml(xml)
    src = read_geotiff(tif, wavelengths=wl)
    src["fwhm"] = fwhm
    if gain is not None and gain.size == wl.size:
        base = src["read_block"]
        g, o = gain, (off if off is not None else np.zeros_like(gain))
        src["read_block"] = lambda r0, r1: base(r0, r1) * g + o
        src["scale"] = 1.0
    src["sensor"] = "EnMAP"
    return src
