"""
parameters.py — the CRISM summary parameters of Viviano-Beck et al. (2014),
hyperspectral formulations, as plain functions f(ctx) -> array.

Formulas and kernel widths follow the PDS Geosciences Node "CRISM Summary
Parameter Calculation Reference" (CAT implementation).

PARAMETERS maps name -> dict(func, wl=[wavelengths needed, nm],
                              group, target)

Departures from CAT, needed so values are comparable across sensors:
  * BDI1000VIS, BDI1000IR, BDI2000 integrate (1 - R/continuum) over
    wavelength in µm (trapezoid). CAT sums over CRISM bands; multiply by
    ~1000/6.55 to compare with CAT numbers.
  * VAR is the mean squared residual × number of CRISM bands in
    1021-2253 nm (189), i.e. CRISM-equivalent regardless of band count.
"""
from __future__ import annotations

import warnings
import numpy as np

from .kernels import (R, RC, RB, W, ratio_to_continuum, bands_in, nan_array,
                      covered)

# Coverage is decided by evaluating each function on a flat spectrum
# (compute.coverage_table); the "wl" lists below are documentation.

# ---------------------------------------------------------------------------
# VNIR
# ---------------------------------------------------------------------------
def R440(c):  return R(c, 440, 5)
def R530(c):  return R(c, 530, 5)
def R600(c):  return R(c, 600, 5)
def R770(c):  return R(c, 770, 5)
def RBR(c):
    with np.errstate(all="ignore"):
        return R(c, 770, 5) / R(c, 440, 5)
def BD530_2(c): return RB(c, 530, 440, 614, 5, 5, 5)
def SH600_2(c):
    with np.errstate(all="ignore"):
        return 1.0 - RC(c, 600, 533, 716, 5, 3) / R(c, 600, 5)
def SH770(c):
    with np.errstate(all="ignore"):
        return 1.0 - RC(c, 775, 716, 860, 3, 5) / R(c, 775, 5)
def BD640_2(c): return RB(c, 624, 600, 760, 3, 5, 5)
def BD860_2(c): return RB(c, 860, 755, 977, 5, 5, 5)
def BD920_2(c): return RB(c, 920, 807, 984, 5, 5, 5)
def IRR1(c):
    with np.errstate(all="ignore"):
        return R(c, 800, 5) / R(c, 997, 5)


def _poly_peak(c):
    """RPEAK1 (µm) and modelled peak reflectance R* (hyperspectral rule)."""
    key = ("RPEAK1",)
    if key in c["cache"]:
        return c["cache"][key]
    wl = c["wl"]
    sel = np.nonzero(((wl >= 410) & (wl <= 644)) | ((wl >= 684) & (wl <= 1023)))[0]
    peak, rstar = nan_array(c), nan_array(c)
    if sel.size >= 8 and covered(c, 442, 900):
        x = wl[sel] / 1000.0
        xm = x.mean()
        V = np.vander(x - xm, 6)                       # 5th-degree
        P = np.linalg.pinv(V)                          # (6, nb)
        Y = c["d"][..., sel].reshape(-1, sel.size)     # (npix, nb)
        ok = np.all(np.isfinite(Y), axis=1)
        grid = np.arange(0.442, 1.0231, 0.002)
        G = np.vander(grid - xm, 6)
        pk = np.full(Y.shape[0], np.nan)
        rs = np.full(Y.shape[0], np.nan)
        idx = np.nonzero(ok)[0]
        for s in range(0, idx.size, 20000):
            ii = idx[s:s + 20000]
            coef = P @ Y[ii].T                         # (6, n)
            F = G @ coef                               # (ngrid, n)
            am = np.argmax(F, axis=0)
            wpk = grid[am]
            # if global max is >= 0.9 µm, take the smallest local maximum
            late = wpk >= 0.9
            if late.any():
                dF = np.diff(F[:, late], axis=0)
                locmax = (dF[:-1] > 0) & (dF[1:] <= 0)
                has = locmax.any(axis=0)
                first = np.argmax(locmax, axis=0) + 1
                cols = np.nonzero(late)[0]
                am[cols[has]] = first[has]
                wpk = grid[am]
            pk[ii] = wpk
            rs[ii] = F[am, np.arange(F.shape[1])]
        peak, rstar = pk.reshape(c["shape"]), rs.reshape(c["shape"])
    c["cache"][key] = (peak, rstar)
    return peak, rstar


def RPEAK1(c):
    return _poly_peak(c)[0]


def _integrate(c, lo, hi, cont, lo_inclusive=True):
    sel = bands_in(c, lo, hi, lo_inclusive)
    if sel.size < 2:
        return nan_array(c)
    x = c["wl"][sel] / 1000.0
    with np.errstate(all="ignore"):
        y = 1.0 - c["d"][..., sel] / cont(x * 1000.0)
    return np.trapezoid(y, x, axis=-1)


def BDI1000VIS(c):
    _, rstar = _poly_peak(c)
    return _integrate(c, 833, 1023, lambda w: rstar[..., None])


# ---------------------------------------------------------------------------
# IR — mafic
# ---------------------------------------------------------------------------
def OLINDEX3(c):
    r1, r2 = R(c, 1750, 7, anchor=True), R(c, 1862, 7, anchor=True)
    w1, w2 = W(c, 1750), W(c, 1862)
    if w2 == w1:
        return nan_array(c)
    slope = (r2 - r1) / (w2 - w1)
    out = 0.0
    for lam, wt in ((1210, .1), (1250, .1), (1263, .2), (1276, .2), (1330, .4)):
        with np.errstate(all="ignore"):
            out = out + wt * (1.0 - R(c, lam, 7) / (r1 + slope * (W(c, lam) - w1)))
    return out


def BD1300(c): return RB(c, 1320, 1080, 1750, 15, 5, 5)


def LCPINDEX2(c):
    return sum(wt * RB(c, l, 1560, 2450, 7, 7, 7)
               for l, wt in ((1690, .2), (1750, .2), (1810, .3), (1870, .3)))


def HCPINDEX2(c):
    return sum(wt * RB(c, l, 1690, 2530, 5 if l == 2120 else 7, 7, 7)
               for l, wt in ((2120, .1), (2140, .1), (2230, .15), (2250, .3),
                             (2430, .2), (2460, .15)))


def _bdicont(c):
    key = ("BDICONT",)
    if key in c["cache"]:
        return c["cache"][key]
    lo, hi = bands_in(c, 1330, 1875), bands_in(c, 2430, 2600)
    if lo.size == 0 or hi.size == 0:
        out = None
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            WL, RL = np.median(c["wl"][lo]), np.nanpercentile(c["d"][..., lo], 75, axis=-1)
            WH, RH = np.median(c["wl"][hi]), np.nanmedian(c["d"][..., hi], axis=-1)
        sl = (RH - RL) / (WH - WL)
        out = lambda w: sl[..., None] * (w - WL) + RL[..., None]
    c["cache"][key] = out
    return out


def BDI1000IR(c):
    f = _bdicont(c)
    return nan_array(c) if f is None else _integrate(c, 1020, 1255, f, lo_inclusive=False)


def BDI2000(c):
    f = _bdicont(c)
    return nan_array(c) if f is None else _integrate(c, 1660, 2390, f)


def VAR(c):
    if not covered(c, 1014, 2287, anchor=True):
        return nan_array(c)
    iL, iH = int(np.argmin(abs(c["wl"] - 1014))), int(np.argmin(abs(c["wl"] - 2287)))
    WL, WH = c["wl"][iL], c["wl"][iH]
    RL, RH = c["d"][..., iL], c["d"][..., iH]
    sel = bands_in(c, 1021, 2253)
    rc = ((RH - RL) / (WH - WL))[..., None] * (c["wl"][sel] - WL) + RL[..., None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean((c["d"][..., sel] - rc) ** 2, axis=-1) * 189.0


def ISLOPE1(c):
    # reflectance per µm
    return (R(c, 1815, 5, anchor=True) - R(c, 2530, 5, anchor=True)) / \
        ((W(c, 2530) - W(c, 1815)) / 1000.0)


# ---------------------------------------------------------------------------
# IR — hydrated / OH / ices
# ---------------------------------------------------------------------------
def BD1400(c):   return RB(c, 1395, 1330, 1467, 3, 5, 5)
def BD1435(c):   return RB(c, 1434, 1370, 1470, 1, 3, 3)
def BD1500_2(c): return RB(c, 1525, 1367, 1808, 11, 5, 5)
def ICER1_2(c):
    with np.errstate(all="ignore"):
        return 1.0 - (1.0 - BD1435(c)) / (1.0 - BD1500_2(c))
def BD1750_2(c): return RB(c, 1750, 1690, 1815, 3, 5, 5)
def BD1900_2(c):
    return 0.5 * RB(c, 1930, 1850, 2067) + 0.5 * RB(c, 1985, 1850, 2067)
def BD1900r2(c):
    num = sum(ratio_to_continuum(c, l, 1815, 2132) for l in (1908, 1914, 1921, 1928, 1934, 1941))
    den = sum(ratio_to_continuum(c, l, 1815, 2132) for l in (1862, 1869, 1875, 2112, 2120, 2126))
    with np.errstate(all="ignore"):
        return 1.0 - num / den
def BD2100_2(c): return RB(c, 2132, 1930, 2250, 5, 3, 3)
def BD2165(c):   return RB(c, 2165, 2120, 2230, 3, 5, 3)
def BD2190(c):   return RB(c, 2185, 2120, 2250, 3, 5, 3)
def MIN2200(c):
    return np.fmin(RB(c, 2165, 2120, 2350, 3, 5, 5), RB(c, 2210, 2120, 2350, 3, 5, 5))
def BD2210_2(c): return RB(c, 2210, 2165, 2290, 5, 5, 5)
def D2200(c):
    f = lambda l, k: ratio_to_continuum(c, l, 1815, 2430, k, 7, 7)
    with np.errstate(all="ignore"):
        return 1.0 - (f(2210, 7) + f(2230, 7)) / (2.0 * f(2165, 5))
def BD2230(c):   return RB(c, 2235, 2210, 2252, 3, 3, 3)
def BD2250(c):   return RB(c, 2245, 2120, 2340, 7, 5, 3)
def MIN2250(c):
    return np.fmin(RB(c, 2210, 2165, 2350, 3, 5, 5), RB(c, 2265, 2165, 2350, 3, 5, 5))
def BD2265(c):   return RB(c, 2265, 2120, 2340, 3, 5, 5)
def BD2290(c):   return RB(c, 2290, 2250, 2350, 5, 5, 5)
def D2300(c):
    f = lambda l, k: ratio_to_continuum(c, l, 1815, 2530, k, 5, 5)
    with np.errstate(all="ignore"):
        return 1.0 - (f(2290, 3) + f(2320, 3) + f(2330, 3)) / (f(2120, 5) + f(2170, 5) + f(2210, 5))
def BD2355(c):   return RB(c, 2355, 2300, 2450, 5, 5, 5)
def SINDEX2(c):
    with np.errstate(all="ignore"):
        return 1.0 - RC(c, 2290, 2120, 2400, 5, 3) / R(c, 2290, 7)
def ICER2_2(c):  return RB(c, 2600, 2456, 2530, 5, 5, 5)
def MIN2295_2480(c):
    return np.fmin(RB(c, 2295, 2165, 2364), RB(c, 2480, 2364, 2570))
def MIN2345_2537(c):
    return np.fmin(RB(c, 2345, 2250, 2430), RB(c, 2537, 2430, 2602))
def BD2500_2(c): return RB(c, 2480, 2364, 2570, 5, 5, 5)
def BD3000(c):
    with np.errstate(all="ignore"):
        return 1.0 - (R(c, 3000, 5) * R(c, 2210, 5)) / (R(c, 2530, 5, anchor=True) ** 2)
def BD3100(c):   return RB(c, 3120, 3000, 3250, 5, 5, 5)
def BD3200(c):   return RB(c, 3320, 3250, 3390, 5, 5, 5)
def BD3400_2(c): return RB(c, 3420, 3250, 3630, 15, 11, 11)
def CINDEX2(c):
    with np.errstate(all="ignore"):
        return 1.0 - RC(c, 3610, 3450, 3875, 9, 7) / R(c, 3610, 11)

# atmospheric / browse components
def R1080(c): return R(c, 1080, 5)
def R1300(c): return R(c, 1300, 5)
def R1506(c): return R(c, 1506, 5)
def R2529(c): return R(c, 2529, 5)
def R3920(c): return R(c, 3920, 5)
def BD2600(c): return RB(c, 2600, 2530, 2630, 5, 5, 5)
def IRR2(c):
    with np.errstate(all="ignore"):
        return R(c, 2530, 5) / R(c, 2210, 5)
def IRR3(c):
    with np.errstate(all="ignore"):
        return R(c, 3500, 7) / R(c, 3390, 7)


# ---------------------------------------------------------------------------
# Registry: name -> func, wavelengths that must be covered, group, target
# ---------------------------------------------------------------------------
def _p(func, wl, group, target):
    return {"func": func, "wl": wl, "group": group, "target": target}


PARAMETERS = {
    # VNIR
    "R440": _p(R440, [440], "VNIR", "reflectance (blue)"),
    "R530": _p(R530, [530], "VNIR", "reflectance (green)"),
    "R600": _p(R600, [600], "VNIR", "reflectance (red)"),
    "R770": _p(R770, [770], "VNIR", "albedo / dust"),
    "RBR": _p(RBR, [440, 770], "VNIR", "red/blue ratio; dust, ferric coatings"),
    "BD530_2": _p(BD530_2, [440, 530, 614], "VNIR", "crystalline ferric minerals"),
    "SH600_2": _p(SH600_2, [533, 600, 716], "VNIR", "select ferric minerals (hematite)"),
    "SH770": _p(SH770, [716, 775, 860], "VNIR", "select ferric (lepidocrocite)"),
    "BD640_2": _p(BD640_2, [600, 624, 760], "VNIR", "select ferric (maghemite)"),
    "BD860_2": _p(BD860_2, [755, 860, 977], "VNIR", "select ferric (hematite)"),
    "BD920_2": _p(BD920_2, [807, 920, 984], "VNIR", "select ferric / low-Ca pyroxene"),
    "RPEAK1": _p(RPEAK1, [442, 900], "VNIR", "reflectance peak position; Fe mineralogy"),
    "BDI1000VIS": _p(BDI1000VIS, [833, 1023], "VNIR", "olivine, pyroxene, Fe glass"),
    "IRR1": _p(IRR1, [800, 997], "VNIR", "IR ratio 1 (aphelion ice clouds / detector)"),
    # IR mafic
    "BDI1000IR": _p(BDI1000IR, [1020, 1255, 1330, 2450], "Mafic", "olivine, pyroxene, Fe glass"),
    "OLINDEX3": _p(OLINDEX3, [1210, 1330, 1750, 1862], "Mafic", "olivine"),
    "BD1300": _p(BD1300, [1080, 1320, 1750], "Mafic", "Fe2+ in plagioclase"),
    "LCPINDEX2": _p(LCPINDEX2, [1560, 1690, 1870, 2450], "Mafic", "low-Ca pyroxene"),
    "HCPINDEX2": _p(HCPINDEX2, [1690, 2120, 2460, 2530], "Mafic", "high-Ca pyroxene"),
    "VAR": _p(VAR, [1014, 2287], "Mafic", "olivine / pyroxene (spectral variance)"),
    "ISLOPE1": _p(ISLOPE1, [1815, 2530], "Mafic", "ferric coatings on dark rock"),
    "BDI2000": _p(BDI2000, [1330, 1660, 2390, 2450], "Mafic", "pyroxene"),
    # hydrated / OH
    "BD1400": _p(BD1400, [1330, 1395, 1467], "Hydrated", "hydrated / hydroxylated minerals"),
    "BD1435": _p(BD1435, [1370, 1434, 1470], "Ices", "CO2 ice; some hydrated minerals"),
    "BD1500_2": _p(BD1500_2, [1367, 1525, 1808], "Ices", "H2O ice"),
    "ICER1_2": _p(ICER1_2, [1370, 1470, 1525, 1808], "Ices", "CO2 vs H2O ice mixtures"),
    "BD1750_2": _p(BD1750_2, [1690, 1750, 1815], "Hydrated", "alunite, gypsum"),
    "BD1900_2": _p(BD1900_2, [1850, 1930, 1985, 2067], "Hydrated", "H2O (bound molecular water)"),
    "BD1900r2": _p(BD1900r2, [1815, 1908, 1941, 2132], "Hydrated", "H2O (bound molecular water)"),
    "BD2100_2": _p(BD2100_2, [1930, 2132, 2250], "Hydrated", "monohydrated sulfates"),
    "BD2165": _p(BD2165, [2120, 2165, 2230], "Hydrated", "kaolin group"),
    "BD2190": _p(BD2190, [2120, 2185, 2250], "Hydrated", "beidellite, allophane"),
    "MIN2200": _p(MIN2200, [2120, 2165, 2210, 2350], "Hydrated", "kaolinite group"),
    "BD2210_2": _p(BD2210_2, [2165, 2210, 2290], "Hydrated", "Al-OH minerals"),
    "D2200": _p(D2200, [1815, 2165, 2230, 2430], "Hydrated", "Al-OH minerals"),
    "BD2230": _p(BD2230, [2210, 2235, 2252], "Hydrated", "hydroxylated ferric sulfate"),
    "BD2250": _p(BD2250, [2120, 2245, 2340], "Hydrated", "opal / hydrated silica"),
    "MIN2250": _p(MIN2250, [2165, 2210, 2265, 2350], "Hydrated", "H-bound Si-OH (opal)"),
    "BD2265": _p(BD2265, [2120, 2265, 2340], "Hydrated", "jarosite, gibbsite, acid-leached nontronite"),
    "BD2290": _p(BD2290, [2250, 2290, 2350], "Hydrated", "Mg/Fe-OH minerals; CO2 ice"),
    "D2300": _p(D2300, [1815, 2120, 2330, 2530], "Hydrated", "hydroxylated Fe/Mg silicates"),
    "BD2355": _p(BD2355, [2300, 2355, 2450], "Hydrated", "chlorite, prehnite, pumpellyite"),
    "SINDEX2": _p(SINDEX2, [2120, 2290, 2400], "Hydrated", "hydrated sulfates (poly vs mono)"),
    "ICER2_2": _p(ICER2_2, [2456, 2530, 2600], "Ices", "CO2 ice"),
    "MIN2295_2480": _p(MIN2295_2480, [2165, 2295, 2480, 2570], "Carbonate", "Mg carbonates"),
    "MIN2345_2537": _p(MIN2345_2537, [2250, 2345, 2537, 2602], "Carbonate", "Fe/Ca carbonates"),
    "BD2500_2": _p(BD2500_2, [2364, 2480, 2570], "Carbonate", "Mg carbonates"),
    "BD3000": _p(BD3000, [2210, 2530, 3000], "Hydrated", "H2O"),
    "BD3100": _p(BD3100, [3000, 3120, 3250], "Ices", "H2O ice"),
    "BD3200": _p(BD3200, [3250, 3320, 3390], "Ices", "CO2 ice"),
    "BD3400_2": _p(BD3400_2, [3250, 3420, 3630], "Carbonate", "carbonates; organics"),
    "CINDEX2": _p(CINDEX2, [3450, 3610, 3875], "Carbonate", "carbonates"),
    # atmospheric / browse components
    "R1080": _p(R1080, [1080], "Reflectance", "IR albedo"),
    "R1300": _p(R1300, [1300], "Reflectance", "IR albedo"),
    "R1506": _p(R1506, [1506], "Reflectance", "IR albedo"),
    "R2529": _p(R2529, [2529], "Reflectance", "IR albedo"),
    "R3920": _p(R3920, [3920], "Reflectance", "3.9 µm reflectance / thermal"),
    "BD2600": _p(BD2600, [2530, 2600, 2630], "Atmosphere", "H2O vapor"),
    "IRR2": _p(IRR2, [2210, 2530], "Atmosphere", "IR ratio 2 (aerosols)"),
    "IRR3": _p(IRR3, [3390, 3500], "Atmosphere", "IR ratio 3"),
}

# Terrestrial-usable subset: avoids H2O-vapour windows (1.35-1.45, 1.8-1.95 µm)
# and the >2.5 µm region that VSWIR sensors do not cover.
TERRESTRIAL = ["R440", "R530", "R600", "R770", "RBR", "BD530_2", "SH600_2",
               "SH770", "BD640_2", "BD860_2", "BD920_2", "RPEAK1", "BDI1000VIS",
               "BDI1000IR", "OLINDEX3", "BD1300", "LCPINDEX2", "HCPINDEX2",
               "BDI2000", "BD1750_2", "BD2100_2", "BD2165", "BD2190", "MIN2200",
               "BD2210_2", "D2200", "BD2230", "BD2250", "MIN2250", "BD2265",
               "BD2290", "D2300", "BD2355", "SINDEX2", "MIN2295_2480",
               "BD2500_2", "R1080", "R1506"]

# Figure 9 column order (Viviano-Beck et al., 2014)
FIG9_COLUMNS = ["OLINDEX3", "BD1300", "LCPINDEX2", "HCPINDEX2", "BD1400", "BD1435",
                "BD1500_2", "BD1750_2", "BD1900r2", "BD2100_2", "BD2165", "BD2190",
                "MIN2200", "BD2210_2", "D2200", "BD2230", "BD2250", "MIN2250",
                "BD2265", "BD2290", "D2300", "BD2355", "SINDEX2", "MIN2295_2480",
                "MIN2345_2537", "BD2500_2"]
