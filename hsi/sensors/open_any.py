"""
open_any.py — one entry point that turns anything spectral into a source dict.

    open_spectral(obj, wavelengths=None, **kw)

obj may be
  numpy array / list           1-D spectrum, 2-D (N, B) table, 3-D cube
                               (bands last, or set band_axis=)
  pandas Series                index = wavelengths
  pandas DataFrame             wide (rows = spectra) or long (cols = spectra)
  xarray DataArray / Dataset
  a source dict                passed through
  a path (str / Path):
    .csv .txt .tab .asc .sed .dat(text) .xlsx .xls .xlsm .ods .parquet   tables
    "refl.txt|wave.txt"                                    USGS splib07
    .sli (+.hdr)                                          ENVI spectral library
    .asd                                                  ASD binary (specdal)
    .hdr / ENVI data file                                 ENVI cube (airborne, lab)
    .tif .tiff .vrt .jp2 .img(GDAL)                       GeoTIFF / GDAL raster
    EMIT_L2A_RFL*.nc                                      EMIT (ortho=True for GLT)
    other .nc / .zarr                                     netCDF via xarray
    PRS_L2*.he5                                           PRISMA
    EnMAP folder / *SPECTRAL_IMAGE*.TIF                   EnMAP L2A
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .source import new_source


def _is_envi_data(p):
    return p.with_suffix(".hdr").exists() or Path(str(p) + ".hdr").exists()


def open_spectral(obj, wavelengths=None, **kw):
    import pandas as pd
    if isinstance(obj, dict) and "kind" in obj and "wl" in obj:
        return obj
    if isinstance(obj, (np.ndarray, list, tuple)):
        if wavelengths is None:
            raise ValueError("arrays need wavelengths=")
        from .tabular_io import from_array
        return from_array(obj, wavelengths, **{k: v for k, v in kw.items()
                                               if k in ("band_axis", "labels", "wl_units")})
    if isinstance(obj, pd.Series):
        from .tabular_io import from_series
        return from_series(obj, kw.get("wl_units"))
    if isinstance(obj, pd.DataFrame):
        from .tabular_io import from_dataframe
        return from_dataframe(obj, kw.get("orientation", "auto"), kw.get("wl_col"),
                              kw.get("id_cols"), kw.get("wl_units"), wavelengths)
    try:
        import xarray as xr
        if isinstance(obj, (xr.DataArray, xr.Dataset)):
            from .xarray_io import from_xarray
            return from_xarray(obj, kw.get("var"), wavelengths, kw.get("wl_coord"),
                               kw.get("wl_units"))
    except ImportError:
        pass

    s = str(obj)
    if "|" in s:
        from .tabular_io import read_usgs_splib
        r, w = s.split("|", 1)
        return read_usgs_splib(r, w)
    p = Path(s)
    name, suf = p.name.upper(), p.suffix.lower()

    if p.is_dir():
        if list(p.glob("*SPECTRAL_IMAGE*")):
            from .spaceborne_io import read_enmap
            return read_enmap(p)
        raise ValueError(f"don't know how to read folder {p}")
    if suf == ".he5" or name.startswith("PRS_"):
        from .spaceborne_io import read_prisma
        return read_prisma(p)
    if suf in (".nc", ".nc4", ".zarr") or s.endswith(".zarr"):
        if name.startswith("EMIT_"):
            from .spaceborne_io import read_emit
            return read_emit(p, ortho=kw.get("ortho", False))
        from .xarray_io import read_netcdf
        return read_netcdf(p, kw.get("var"), wavelengths)
    if "SPECTRAL_IMAGE" in name and suf in (".tif", ".tiff"):
        from .spaceborne_io import read_enmap
        return read_enmap(p)
    if suf == ".sli":
        from .tabular_io import read_envi_sli
        return read_envi_sli(p)
    if suf == ".asd":
        from .tabular_io import read_asd
        return read_asd(p)
    if suf in (".xlsx", ".xls", ".xlsm", ".ods", ".parquet", ".csv", ".tab", ".asc", ".sed",
               ".txt", ".tsv"):
        from .tabular_io import read_table
        return read_table(p, kw.get("sheet"), kw.get("orientation", "auto"), kw.get("wl_col"),
                          kw.get("id_cols"), kw.get("wl_units"))
    if suf == ".hdr" or _is_envi_data(p):
        from .envi_io import read_envi
        return read_envi(p, wavelengths, kw.get("wl_units"))
    if suf in (".tif", ".tiff", ".vrt", ".jp2", ".img"):
        from .geotiff_io import read_geotiff
        return read_geotiff(p, wavelengths, kw.get("wl_units"), kw.get("xml"))
    if suf == ".dat":
        from .tabular_io import read_table
        return read_table(p)
    raise ValueError(f"unrecognised input: {obj!r}")
