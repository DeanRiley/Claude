"""
outputs.py — block-streaming writers for parameter images and readers back.

open_writer(path, fmt, rows, cols, names, crs, transform) returns a dict of
closures {"write": f(r0, r1, {name: block}), "close": f()}.

Formats (from extension or fmt=): GeoTIFF (.tif), ENVI (.img/.hdr/.dat),
netCDF (.nc), NumPy (.npz). GeoTIFF and ENVI stream to disk; netCDF and
npz accumulate in memory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _fmt(path, fmt):
    if fmt:
        return fmt.lower()
    s = Path(path).suffix.lower()
    return {".tif": "gtiff", ".tiff": "gtiff", ".img": "envi", ".hdr": "envi", ".dat": "envi",
            ".nc": "netcdf", ".npz": "npz"}.get(s, "gtiff")


def open_writer(path, fmt, rows, cols, names, crs=None, transform=None, wl_use=None):
    fmt = _fmt(path, fmt)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "gtiff":
        import rasterio
        from rasterio.windows import Window
        prof = dict(driver="GTiff", height=rows, width=cols, count=len(names),
                    dtype="float32", nodata=np.nan, compress="deflate", predictor=3,
                    tiled=True, blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
        if cols < 256 or rows < 256:
            prof.update(tiled=False)
            prof.pop("blockxsize"); prof.pop("blockysize")
        if crs is not None:
            prof["crs"] = crs
        if transform is not None:
            prof["transform"] = transform
        ds = rasterio.open(str(path), "w", **prof)
        for i, n in enumerate(names, 1):
            ds.set_band_description(i, n)
        ds.update_tags(source="CRISM summary parameters (Viviano-Beck et al. 2014)",
                       parameters=",".join(names))

        def write(r0, r1, res):
            arr = np.stack([res[n] for n in names]).astype(np.float32)
            ds.write(arr, window=Window(0, r0, cols, r1 - r0))
        return {"write": write, "close": ds.close}

    if fmt == "envi":
        import spectral.io.envi as envi
        hdr = path.with_suffix(".hdr")
        md = {"band names": list(names), "data ignore value": "nan",
              "description": "CRISM summary parameters (Viviano-Beck et al. 2014)"}
        if transform is not None and crs is not None:
            t = transform
            md["map info"] = ["Arbitrary", "1", "1", str(t.c), str(t.f), str(t.a), str(-t.e),
                              "units=Meters"]
            md["coordinate system string"] = [crs.to_wkt()]
        img = envi.create_image(str(hdr), md, shape=(rows, cols, len(names)),
                                dtype=np.float32, interleave="bsq", force=True)
        mm = img.open_memmap(writable=True)

        def write(r0, r1, res):
            mm[r0:r1] = np.stack([res[n] for n in names], -1).astype(np.float32)

        def close():
            mm.flush()
        return {"write": write, "close": close}

    # in-memory formats
    store = {n: np.full((rows, cols), np.nan, np.float32) for n in names}

    def write(r0, r1, res):
        for n in names:
            store[n][r0:r1] = res[n]

    def close():
        if fmt == "npz":
            np.savez_compressed(str(path), **store)
        else:
            import xarray as xr
            ds = xr.Dataset({n: (("y", "x"), v) for n, v in store.items()})
            if transform is not None:
                t = transform
                ds = ds.assign_coords(x=t.c + t.a * (np.arange(cols) + 0.5),
                                      y=t.f + t.e * (np.arange(rows) + 0.5))
            if crs is not None:
                ds.attrs["crs_wkt"] = crs.to_wkt()
            ds.attrs["source"] = "CRISM summary parameters (Viviano-Beck et al. 2014)"
            ds.to_netcdf(str(path))
    return {"write": write, "close": close}


def read_params(path, names=None):
    """Read a parameter image written by open_writer back into {name: 2-D}."""
    fmt = _fmt(path, None)
    if fmt == "gtiff":
        import rasterio
        with rasterio.open(str(path)) as ds:
            desc = list(ds.descriptions)
            names = names or desc
            return {n: ds.read(desc.index(n) + 1).astype(float) for n in names}
    if fmt == "envi":
        import spectral.io.envi as envi
        img = envi.open(str(Path(path).with_suffix(".hdr")))
        bn = img.metadata["band names"]
        names = names or bn
        return {n: np.asarray(img.read_band(bn.index(n)), float) for n in names}
    if fmt == "npz":
        z = np.load(str(path))
        return {n: z[n].astype(float) for n in (names or z.files)}
    import xarray as xr
    ds = xr.open_dataset(str(path))
    return {n: ds[n].values.astype(float) for n in (names or list(ds.data_vars))}
