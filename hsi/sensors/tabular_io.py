"""
tabular_io.py — 1-D / 2-D spectra: numpy arrays, lists, pandas Series and
DataFrames, CSV / TXT / TAB / Excel / ODS files, USGS splib07 ASCII,
ENVI spectral libraries (.sli), ASD binary (.asd, via `specdal` if present),
Spectral Evolution .sed and other "header + numeric block" text exports.

DataFrame layouts recognised automatically
------------------------------------------
wide  : one spectrum per ROW, wavelength column names ("450", "450.5 nm",
        "R_450", "b450", "2.205") + any non-spectral columns (IDs, lat/lon,
        assays...) which are carried through to the output as metadata.
long  : one spectrum per COLUMN, wavelengths in the index or in a column
        named wavelength / wl / wave / lambda / nm / um (or the first
        numeric monotonic column).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .source import new_source, to_nm

_WL_NAMES = {"wavelength", "wavelengths", "wl", "wave", "wavelength_nm", "wavelength_um",
             "lambda", "nm", "um", "µm", "wvl", "wavelength (nm)", "wavelength (um)",
             "wavelength(nm)", "wavelength(um)", "wavelength_(nm)"}


def _name_to_wl(name):
    """'450' -> 450, 'R_450.5nm' -> 450.5, '2.205um' -> 2205; None if no match."""
    if isinstance(name, (int, float, np.integer, np.floating)):
        return float(name)
    s = str(name).strip()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(nm|um|µm|μm)?\s*\)?\s*$", s, re.IGNORECASE)
    if not m:
        return None
    v = float(m.group(1))
    unit = (m.group(2) or "").lower()
    if unit in ("um", "µm", "μm"):
        v *= 1000.0
    return v


def _plausible_wl(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size < 5:
        return False
    nm = to_nm(v)
    return (nm.min() >= 250) and (nm.max() <= 20000) and np.all(np.diff(np.sort(nm)) >= 0)


# ---------------------------------------------------------------------------
# arrays / Series / DataFrames
# ---------------------------------------------------------------------------
def from_array(arr, wavelengths, band_axis=None, labels=None, wl_units=None):
    """1-D (B,), 2-D (N,B) -> spectra; 3-D (rows, cols, B) -> image.
    band_axis: axis holding bands if not last (e.g. 0 for rasterio (B,R,C))."""
    a = np.asarray(arr, float)
    wl = to_nm(np.asarray(wavelengths, float), wl_units)
    if band_axis is None:
        if a.shape[-1] != wl.size and a.shape[0] == wl.size:
            band_axis = 0
        else:
            band_axis = a.ndim - 1
    a = np.moveaxis(a, band_axis, -1)
    if a.shape[-1] != wl.size:
        raise ValueError(f"no axis of length {wl.size} (wavelengths) in array {arr.shape}")
    if a.ndim == 1:
        return new_source("spectra", wl, data=a[None, :], labels=labels or ["spectrum"],
                          meta=None, squeeze=True)
    if a.ndim == 2:
        labels = labels or [f"spectrum_{i}" for i in range(a.shape[0])]
        return new_source("spectra", wl, data=a, labels=list(labels))
    if a.ndim == 3:
        return new_source("image", wl, data=a)
    raise ValueError("arrays must be 1-D, 2-D or 3-D")


def from_series(s, wl_units=None):
    wl = np.array([_name_to_wl(i) for i in s.index], dtype=float)
    keep = np.isfinite(wl)
    return new_source("spectra", to_nm(wl[keep], wl_units),
                      data=pd.to_numeric(s[keep], errors="coerce").to_numpy(float)[None, :],
                      labels=[s.name if s.name is not None else "spectrum"], squeeze=True)


def from_dataframe(df, orientation="auto", wl_col=None, id_cols=None, wl_units=None,
                   wavelengths=None):
    """orientation: 'auto' | 'wide' (rows = spectra) | 'long' (columns = spectra)."""
    df = df.copy()
    if wavelengths is not None:            # explicit: columns given are spectral
        wl = to_nm(np.asarray(wavelengths, float), wl_units)
        spec_cols = [c for c in df.columns if c not in (id_cols or [])][:wl.size]
        return _wide(df, spec_cols, wl, id_cols)

    if orientation in ("auto", "long"):
        col = wl_col
        if col is None:
            for c in df.columns:
                if str(c).strip().lower() in _WL_NAMES:
                    col = c
                    break
        if col is not None:
            return _long(df.set_index(col), wl_units, units_hint=str(col))
        if orientation == "long" or (_plausible_wl(pd.to_numeric(pd.Index(df.index), errors="coerce"))
                                     and not _wide_candidates(df)[0]):
            if orientation == "long" and not _plausible_wl(pd.to_numeric(pd.Index(df.index), errors="coerce")):
                first = df.columns[0]
                return _long(df.set_index(first), wl_units, units_hint=str(first))
            return _long(df, wl_units)

    cols, wl = _wide_candidates(df)
    if cols:
        return _wide(df, cols, to_nm(np.array(wl), wl_units), id_cols)
    # last try: first column looks like wavelengths -> long
    first = df.columns[0]
    if _plausible_wl(pd.to_numeric(df[first], errors="coerce")):
        return _long(df.set_index(first), wl_units, units_hint=str(first))
    raise ValueError("could not find wavelengths in DataFrame; pass wavelengths= or wl_col=")


def _wide_candidates(df):
    cols, wl = [], []
    for c in df.columns:
        v = _name_to_wl(c)
        if v is not None and np.issubdtype(pd.to_numeric(df[c], errors="coerce").dtype, np.number) \
                and pd.to_numeric(df[c], errors="coerce").notna().any():
            cols.append(c)
            wl.append(v)
    if len(cols) < 10 or not _plausible_wl(wl):
        return [], []
    return cols, wl


def _wide(df, spec_cols, wl_nm, id_cols=None):
    data = df[spec_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    meta_cols = id_cols if id_cols is not None else [c for c in df.columns if c not in spec_cols]
    meta = df[meta_cols].reset_index(drop=False) if meta_cols else df.index.to_frame(index=False)
    labels = (df[meta_cols[0]].astype(str).tolist() if meta_cols else
              [str(i) for i in df.index])
    return new_source("spectra", wl_nm, data=data, meta=meta, labels=labels)


def _long(df, wl_units=None, units_hint=""):
    idx = pd.to_numeric(pd.Index(df.index), errors="coerce").to_numpy(float)
    keep = np.isfinite(idx)
    df = df.loc[keep]
    num = df.apply(pd.to_numeric, errors="coerce")
    num = num.loc[:, num.notna().any()]
    if wl_units is None and re.search(r"\b(um|µm|micro)", units_hint, re.I):
        wl_units = "um"
    wl = to_nm(idx[keep], wl_units)
    labels = [str(c) for c in num.columns]
    meta = pd.DataFrame({"label": labels})
    return new_source("spectra", wl, data=num.to_numpy(float).T, meta=meta, labels=labels)


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------
def read_text_block(path):
    """Robust reader for text exports with arbitrary headers (ASD txt, SED,
    USGS, RELAB, CSV/TSV). Returns DataFrame of the largest numeric block,
    with column names from the line preceding it when available."""
    lines = Path(path).read_text(errors="ignore").splitlines()
    split = lambda s: [t for t in re.split(r"[,\t;]+|\s{1,}", s.strip()) if t != ""]

    def isnum(t):
        try:
            float(t)
            return True
        except ValueError:
            return False

    best, start, n = (0, 0, 0), None, 0
    for i, ln in enumerate(lines + [""]):
        toks = split(ln)
        ok = len(toks) >= 1 and all(isnum(t) for t in toks)
        if ok:
            if start is None:
                start, n = i, len(toks)
            elif len(toks) != n:
                if i - start > best[1] - best[0]:
                    best = (start, i, n)
                start, n = i, len(toks)
        else:
            if start is not None and i - start > best[1] - best[0]:
                best = (start, i, n)
            start = None
    s, e, n = best
    if e - s < 5:
        raise ValueError(f"no numeric block found in {path}")
    arr = np.array([[float(t) for t in split(l)][:n] for l in lines[s:e]])
    names = None
    if s > 0:
        hdr = [t for t in re.split(r"[,\t;]+|\s{2,}", lines[s - 1].strip()) if t]
        if len(hdr) == n:
            names = hdr
    return pd.DataFrame(arr, columns=names)


def read_table(path, sheet=None, orientation="auto", wl_col=None, id_cols=None,
               wl_units=None, **read_kw):
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".xlsx", ".xlsm", ".xls", ".ods"):
        if sheet is None:
            sheets = pd.read_excel(p, sheet_name=None, **read_kw)
            srcs = []
            for name, df in sheets.items():
                if df.dropna(how="all").shape[0] < 2:
                    continue
                try:
                    s = from_dataframe(df.dropna(how="all", axis=1), orientation, wl_col,
                                       id_cols, wl_units)
                    s["meta"] = s["meta"].assign(sheet=name)
                    srcs.append(s)
                except ValueError:
                    continue
            if not srcs:
                raise ValueError(f"no spectral sheet found in {p}")
            return merge_spectra(srcs, path=str(p))
        df = pd.read_excel(p, sheet_name=sheet, **read_kw)
        s = from_dataframe(df.dropna(how="all", axis=1), orientation, wl_col, id_cols, wl_units)
    elif suf == ".parquet":
        s = from_dataframe(pd.read_parquet(p), orientation, wl_col, id_cols, wl_units)
    else:
        try:
            df = pd.read_csv(p, sep=None, engine="python", comment="#", **read_kw)
            s = from_dataframe(df, orientation, wl_col, id_cols, wl_units)
        except Exception:
            df = read_text_block(p)
            if df.columns.dtype == object:
                s = from_dataframe(df, "long", wl_col or df.columns[0], id_cols, wl_units)
            else:
                df.columns = ["wavelength"] + [f"{p.stem}" if df.shape[1] == 2 else f"{p.stem}_{i}"
                                               for i in range(1, df.shape[1])]
                s = from_dataframe(df, "long", "wavelength", id_cols, wl_units)
    s["path"] = str(p)
    return s


def read_usgs_splib(refl_file, wave_file, label=None):
    """USGS splib07 ASCII: one value per line after a 1-line header."""
    r = read_text_block(refl_file).iloc[:, 0].to_numpy(float)
    w = read_text_block(wave_file).iloc[:, 0].to_numpy(float)
    n = min(r.size, w.size)
    r = np.where(r[:n] < -1e30, np.nan, r[:n])
    return new_source("spectra", to_nm(w[:n]), data=r[None, :],
                      labels=[label or Path(refl_file).stem], path=str(refl_file))


def read_envi_sli(path):
    import spectral.io.envi as envi
    p = Path(path)
    hdr = p if p.suffix.lower() == ".hdr" else p.with_suffix(".hdr")
    lib = envi.open(str(hdr), str(p) if p.suffix.lower() != ".hdr" else None)
    md = lib.metadata
    wl = to_nm(np.asarray(lib.bands.centers, float), md.get("wavelength units"))
    names = list(lib.names) if lib.names else [f"spec_{i}" for i in range(lib.spectra.shape[0])]
    data = np.asarray(lib.spectra, float)
    if "reflectance scale factor" in md:
        data = data / float(md["reflectance scale factor"])
    return new_source("spectra", wl, data=data, labels=names,
                      meta=pd.DataFrame({"label": names}), path=str(p), scale=1.0)


def read_asd(path):
    try:
        from specdal import Spectrum  # optional dependency
    except ImportError as e:
        raise ImportError("binary .asd files need `pip install specdal` "
                          "(or export to text from ViewSpec / RS3)") from e
    sp = Spectrum(filepath=str(path))
    m = sp.measurement
    return new_source("spectra", m.index.to_numpy(float), data=m.to_numpy(float)[None, :],
                      labels=[Path(path).stem], path=str(path))


def merge_spectra(srcs, path=""):
    """Stack several spectra sources; resamples onto the first one's
    wavelengths by linear interpolation if grids differ."""
    wl0 = srcs[0]["wl"]
    data, labels, metas = [], [], []
    for s in srcs:
        d = s["data"]
        if s["wl"].size != wl0.size or not np.allclose(s["wl"], wl0):
            d = np.vstack([np.interp(wl0, s["wl"], row, left=np.nan, right=np.nan) for row in d])
        data.append(d)
        labels += list(s["labels"])
        metas.append(s["meta"] if s["meta"] is not None else pd.DataFrame({"label": s["labels"]}))
    return new_source("spectra", wl0, data=np.vstack(data), labels=labels,
                      meta=pd.concat(metas, ignore_index=True), path=path)
