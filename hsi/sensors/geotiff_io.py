"""
geotiff_io.py — multiband GeoTIFF / any GDAL raster via rasterio.

Wavelengths are looked for, in order:
  1. wavelengths= argument
  2. per-band tags ("wavelength", "WAVELENGTH", "center_wavelength")
  3. ENVI metadata domain ("wavelength" list) written by GDAL's ENVI driver
  4. band descriptions containing a number ("Band 12 (550.3 nm)", "0.5503")
  5. a companion ENVI header (<stem>.hdr) or sensor XML (EnMAP/DESIS style)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .source import new_source, to_nm


def _parse_list(s):
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)]


def wavelengths_from_rasterio(ds):
    nb = ds.count
    units = None
    # 2. per-band tags
    wl = []
    for b in range(1, nb + 1):
        t = {k.lower(): v for k, v in ds.tags(b).items()}
        v = t.get("wavelength") or t.get("center_wavelength") or t.get("wavelength_nm")
        units = units or t.get("wavelength_units") or t.get("units")
        if v is None:
            wl = None
            break
        wl.append(float(_parse_list(v)[0]))
    if wl:
        return to_nm(np.array(wl), units)
    # 3. ENVI domain
    try:
        envi_md = {k.lower(): v for k, v in ds.tags(ns="ENVI").items()}
        if "wavelength" in envi_md:
            v = _parse_list(envi_md["wavelength"])
            if len(v) == nb:
                return to_nm(np.array(v), envi_md.get("wavelength_units"))
    except Exception:
        pass
    # 4. descriptions
    desc = ds.descriptions
    if desc and all(desc):
        vals = []
        for d in desc:
            m = re.findall(r"(\d+\.?\d*)\s*(nm|um|µm)?", d)
            if not m:
                vals = None
                break
            num, unit = m[-1]
            vals.append(float(num) * (1000 if unit in ("um", "µm") else 1))
        if vals:
            return to_nm(np.array(vals))
    return None


def wavelengths_from_xml(xml_path):
    """Generic sensor-XML parser (EnMAP, DESIS, others): returns wl, fwhm,
    gain, offset arrays (gain/offset may be None)."""
    txt = Path(xml_path).read_text(errors="ignore")
    def grab(*tags):
        for t in tags:
            v = re.findall(rf"<{t}[^>]*>\s*([-+0-9.eE]+)\s*</{t}>", txt)
            if v:
                return np.array(v, float)
        return None
    wl = grab("wavelengthCenterOfBand", "centerWavelength", "CentralWavelength", "wavelength")
    fwhm = grab("FWHMOfBand", "fwhm", "FWHM", "bandwidth")
    gain = grab("GainOfBand", "gain")
    off = grab("OffsetOfBand", "offset")
    return wl, fwhm, gain, off


def read_geotiff(path, wavelengths=None, wl_units=None, xml=None, nodata=None):
    import rasterio
    from rasterio.windows import Window
    p = Path(path)
    ds = rasterio.open(str(p))
    rows, cols, nb = ds.height, ds.width, ds.count
    fwhm, scale = None, None
    if wavelengths is not None:
        wl = to_nm(np.asarray(wavelengths, float), wl_units)
    else:
        wl = wavelengths_from_rasterio(ds)
        if wl is None:
            hdr = p.with_suffix(".hdr")
            xmls = [Path(xml)] if xml else sorted(p.parent.glob("*METADATA*.XML")) + \
                sorted(p.parent.glob("*METADATA*.xml"))
            if hdr.exists():
                from spectral.io.envi import read_envi_header
                md = read_envi_header(str(hdr))
                wl = to_nm(np.asarray(md["wavelength"], float), md.get("wavelength units"))
            elif xmls:
                wl, fwhm, gain, off = wavelengths_from_xml(xmls[0])
                if gain is not None and gain.size == nb and np.allclose(gain, gain[0]):
                    scale = float(gain[0])
        if wl is None or len(wl) != nb:
            raise ValueError(f"cannot find {nb} wavelengths for {p}; pass wavelengths=")
    nd = nodata if nodata is not None else ds.nodata

    def read_block(r0, r1, _ds=ds):
        blk = _ds.read(window=Window(0, r0, cols, r1 - r0)).astype(float)
        if nd is not None:
            blk[blk == nd] = np.nan
        return np.moveaxis(blk, 0, -1)

    return new_source("image", wl, fwhm=fwhm, read_block=read_block, shape=(rows, cols, nb),
                      scale=scale, crs=ds.crs, transform=ds.transform,
                      sensor="GeoTIFF", path=str(p))
