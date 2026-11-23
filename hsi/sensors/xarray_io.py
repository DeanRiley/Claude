"""
xarray_io.py — xarray DataArray / Dataset (incl. dask-backed) and generic
netCDF / Zarr files that are not EMIT.

The spectral dimension is found by name (wavelength, wl, band, bands,
lambda, spectral) or by a coordinate whose values look like wavelengths.
1 other dimension -> spectra; 2 other dimensions -> image.
"""
from __future__ import annotations

import numpy as np

from .source import new_source, to_nm

_SPEC_DIMS = ("wavelength", "wavelengths", "wl", "band", "bands", "lambda",
              "spectral", "spectral_band", "bands_wavelength")


def _spectral_dim(da, wl_coord=None):
    if wl_coord:
        return wl_coord
    for d in da.dims:
        if str(d).lower() in _SPEC_DIMS:
            return d
    for d in da.dims:
        if d in da.coords:
            v = np.asarray(da[d].values)
            if np.issubdtype(v.dtype, np.number) and v.size > 5:
                nm = to_nm(v.astype(float))
                if 250 <= np.nanmin(nm) and np.nanmax(nm) <= 20000:
                    return d
    raise ValueError(f"no spectral dimension among {da.dims}")


def from_xarray(obj, var=None, wavelengths=None, wl_coord=None, wl_units=None):
    import xarray as xr
    if isinstance(obj, xr.Dataset):
        if var is None:
            cands = [v for v in ("reflectance", "Reflectance", "rfl", "refl", "data")
                     if v in obj.data_vars]
            var = cands[0] if cands else max(obj.data_vars, key=lambda v: obj[v].ndim)
        da = obj[var]
    else:
        da = obj
    sd = _spectral_dim(da, wl_coord)
    if wavelengths is not None:
        wl = to_nm(np.asarray(wavelengths, float), wl_units)
    else:
        coord = da[sd].values if sd in da.coords else None
        for name in ("wavelength", "wavelengths", "wl", "center_wavelength"):
            if name in da.coords and da[name].dims == (sd,):
                coord = da[name].values
                break
        if coord is None or not np.issubdtype(np.asarray(coord).dtype, np.number):
            raise ValueError("pass wavelengths= (spectral coordinate is not numeric)")
        units = wl_units or (da[sd].attrs.get("units") if sd in da.coords else None)
        wl = to_nm(np.asarray(coord, float), units)
    others = [d for d in da.dims if d != sd]
    da = da.transpose(*others, sd)
    if len(others) == 1:
        labels = [str(v) for v in (da[others[0]].values if others[0] in da.coords
                                   else range(da.sizes[others[0]]))]
        return new_source("spectra", wl, data=np.asarray(da.values, float), labels=labels)
    if len(others) != 2:
        raise ValueError("expected 1 or 2 non-spectral dimensions")
    rows, cols = da.sizes[others[0]], da.sizes[others[1]]
    read_block = lambda r0, r1: np.asarray(da.isel({others[0]: slice(r0, r1)}).values, float)
    crs = transform = None
    try:
        import rioxarray  # noqa: F401
        crs, transform = da.rio.crs, da.rio.transform()
    except Exception:
        pass
    return new_source("image", wl, read_block=read_block, shape=(rows, cols, wl.size),
                      crs=crs, transform=transform, sensor="xarray")


def read_netcdf(path, var=None, wavelengths=None):
    import xarray as xr
    import netCDF4
    with netCDF4.Dataset(str(path)) as nc:
        if "sensor_band_parameters" in nc.groups and "reflectance" in nc.variables:
            from .spaceborne_io import read_emit
            return read_emit(path)
    ds = xr.open_zarr(str(path)) if str(path).endswith(".zarr") else xr.open_dataset(str(path))
    src = from_xarray(ds, var=var, wavelengths=wavelengths)
    src["path"] = str(path)
    return src
