"""
test_crism_indices.py — end-to-end tests over every supported input type.

    python -m pytest hsi/tests/test_crism_indices.py -q
    python hsi/tests/test_crism_indices.py           (runs all, prints summary)

Synthetic spectra only (hsi/tests/synthetic_spectra.py); they check plumbing
and the sign/location of responses, not absolute values.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hsi.tests.synthetic_spectra import demo_spectrum                       # noqa: E402
from hsi.spectral_indices.crism.compute import (compute, coverage_table,    # noqa: E402
                                                CRISM_IR_WL)
from hsi.spectral_indices.crism.outputs import read_params                   # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="crism_test_"))

SENSORS = {
    "CRISM_IR": CRISM_IR_WL,
    "ASD_lab_1nm": np.arange(350.0, 2501.0, 1.0),
    "AVIRIS-NG": np.arange(380.0, 2511.0, 5.0),
    "EMIT": np.linspace(381.0, 2493.0, 285),
    "PRISMA": np.linspace(402.0, 2497.0, 230),
    "EnMAP": np.linspace(418.0, 2445.0, 224),
    "HyMap": np.arange(450.0, 2481.0, 15.0),
    "AISA_Hawk": np.arange(970.0, 2500.0, 6.3),
}
PATCHES = ["Kaolinite", "Mg-Smectite", "Alunite", "Calcite", "Magnesite",
           "Fe-Olivine", "Low-Ca Pyroxene", "Hydrated Silica"]


def make_cube(wl, rows=40, cols=48, dn=True):
    rng = np.random.default_rng(1)
    cube = np.empty((rows, cols, wl.size))
    truth = np.empty((rows, cols), dtype=object)
    bw = cols // 4
    bh = rows // 2
    for k, m in enumerate(PATCHES):
        r0, c0 = (k // 4) * bh, (k % 4) * bw
        spec = demo_spectrum(m, wl, rng)
        cube[r0:r0 + bh, c0:c0 + bw] = spec + rng.normal(0, 0.002, (bh, bw, wl.size))
        truth[r0:r0 + bh, c0:c0 + bw] = m
    if dn:
        cube = np.round(cube * 10000).astype(np.int16)
    return cube, truth


def patch_mean(img, truth, m):
    return float(np.nanmean(img[truth == m]))


# ---------------------------------------------------------------------------
# 1-D / 2-D spectra
# ---------------------------------------------------------------------------
def test_1d_numpy_all_sensors():
    for name, wl in SENSORS.items():
        s = compute(demo_spectrum("Kaolinite", wl), wavelengths=wl, verbose=False)
        assert isinstance(s, pd.Series)
        assert s["BD2165"] > 0.03 and s["D2200"] > 0.03, (name, s["BD2165"], s["D2200"])
        assert s["MIN2200"] > 0.03, name


def test_carbonate_response():
    for name, wl in SENSORS.items():
        if wl.max() < 2570:
            continue
        s = compute(demo_spectrum("Magnesite", wl), wavelengths=wl)
        assert s["MIN2295_2480"] > 0.05, name


def test_kernel_modes_and_resampling_agree():
    wl = SENSORS["ASD_lab_1nm"]
    spec = demo_spectrum("Kaolinite", wl)
    a = compute(spec, wavelengths=wl, names="fig9")
    b = compute(spec, wavelengths=wl, names="fig9", resample_to="crism")
    keys = ["BD2165", "BD2210_2", "D2200", "MIN2200", "BD1900r2"]
    assert np.allclose(a[keys].astype(float), b[keys].astype(float), atol=0.03), (a[keys], b[keys])


def test_coverage():
    cov = coverage_table(SENSORS["AVIRIS-NG"]).set_index("parameter")
    assert cov.loc["BD2165", "covered"] and cov.loc["BD530_2", "covered"]
    assert not cov.loc["CINDEX2", "covered"] and not cov.loc["BD3100", "covered"]
    cov = coverage_table(CRISM_IR_WL).set_index("parameter")
    assert cov.loc["CINDEX2", "covered"] and not cov.loc["BD530_2", "covered"]


def _library(wl):
    return np.vstack([demo_spectrum(m, wl) for m in PATCHES])


def test_2d_numpy_and_dataframes():
    wl = SENSORS["ASD_lab_1nm"]
    lib = _library(wl)
    df = compute(lib, wavelengths=wl, labels=PATCHES)
    assert list(df["label"]) == PATCHES
    # wide DataFrame with metadata columns and "450.0 nm"-style names
    wide = pd.DataFrame(lib * 100, columns=[f"{w:.1f} nm" for w in wl])  # percent
    wide.insert(0, "sample", PATCHES)
    wide.insert(1, "SiO2_wt%", np.linspace(45, 75, len(PATCHES)))
    d2 = compute(wide)
    assert "SiO2_wt%" in d2.columns and np.allclose(d2["BD2165"], df["BD2165"], atol=1e-6)
    # long DataFrame (columns = spectra, wavelength column in µm)
    long = pd.DataFrame(lib.T, columns=PATCHES)
    long.insert(0, "Wavelength (um)", wl / 1000)
    d3 = compute(long)
    assert np.allclose(d3["BD2165"], df["BD2165"], atol=1e-6)
    # Series
    s = pd.Series(lib[0], index=wl, name="kao")
    assert abs(compute(s)["BD2165"] - df["BD2165"][0]) < 1e-9


def test_files_excel_csv_text_usgs_sli():
    wl = SENSORS["ASD_lab_1nm"]
    lib = _library(wl)
    ref = compute(lib, wavelengths=wl, labels=PATCHES)["BD2165"].to_numpy()
    # Excel: one wide sheet, one long sheet, one junk sheet
    xl = TMP / "spectra.xlsx"
    with pd.ExcelWriter(xl) as w:
        wide = pd.DataFrame(lib, columns=wl)
        wide.insert(0, "ID", PATCHES)
        wide.to_excel(w, sheet_name="wide", index=False)
        long = pd.DataFrame(lib.T, columns=[p + "_b" for p in PATCHES])
        long.insert(0, "wavelength", wl)
        long.to_excel(w, sheet_name="long", index=False)
        pd.DataFrame({"note": ["field notes"]}).to_excel(w, sheet_name="notes", index=False)
    d = compute(xl)
    assert len(d) == 2 * len(PATCHES) and set(d["sheet"]) == {"wide", "long"}
    assert np.allclose(d["BD2165"].to_numpy()[:8], ref) and np.allclose(d["BD2165"].to_numpy()[8:], ref)
    d = compute(xl, sheet="long")
    assert np.allclose(d["BD2165"], ref)
    # CSV
    csvp = TMP / "spectra.csv"
    long.to_csv(csvp, index=False)
    assert np.allclose(compute(csvp)["BD2165"], ref)
    # SED / ASD-style text with header block
    sed = TMP / "kao.sed"
    body = "\n".join(f"{w:.1f}\t{r*100:.4f}" for w, r in zip(wl, lib[0]))
    sed.write_text("Version: 2.3\nInstrument: PSR+\nUnits: Reflectance %\nData:\nWvl\tReflect. %\n" + body)
    assert abs(compute(sed)["BD2165"].iloc[0] - ref[0]) < 1e-3
    # USGS splib07 pair
    (TMP / "usgs_refl.txt").write_text("splib07a Kaolinite\n" + "\n".join(f"{r:.6f}" for r in lib[0]))
    (TMP / "usgs_wave.txt").write_text("Wavelengths ASD\n" + "\n".join(f"{w/1000:.5f}" for w in wl))
    assert abs(compute(f"{TMP/'usgs_refl.txt'}|{TMP/'usgs_wave.txt'}")["BD2165"].iloc[0] - ref[0]) < 1e-3
    # ENVI spectral library
    import spectral.io.envi as envi
    from spectral.io.envi import SpectralLibrary
    lib_obj = SpectralLibrary(lib.astype(np.float32),
                              {"wavelength": list(wl / 1000), "wavelength units": "Micrometers",
                               "spectra names": PATCHES}, {})
    lib_obj.save(str(TMP / "lib"))
    d = compute(TMP / "lib.sli")
    assert np.allclose(d["BD2165"], ref, atol=1e-4)


# ---------------------------------------------------------------------------
# images
# ---------------------------------------------------------------------------
def _check_image(params, truth):
    p = params
    assert patch_mean(p["BD2165"], truth, "Kaolinite") > 0.03
    if "D2300" in p and np.isfinite(p["D2300"]).any():
        assert patch_mean(p["D2300"], truth, "Mg-Smectite") > 0.03
    else:
        assert patch_mean(p["BD2290"], truth, "Mg-Smectite") > 0.03
    if "BD1750_2" in p and np.isfinite(p["BD1750_2"]).any():
        assert patch_mean(p["BD1750_2"], truth, "Alunite") > 0.03
    if "OLINDEX3" in p and np.isfinite(p["OLINDEX3"]).any():
        assert patch_mean(p["OLINDEX3"], truth, "Fe-Olivine") > patch_mean(p["OLINDEX3"], truth, "Kaolinite")


def test_image_numpy_and_xarray():
    import xarray as xr
    wl = SENSORS["AVIRIS-NG"]
    cube, truth = make_cube(wl)
    r = compute(cube, wavelengths=wl, verbose=False)          # DN int16 -> auto x1e-4
    _check_image(r["params"], truth)
    # bands-first (rasterio order)
    r2 = compute(np.moveaxis(cube, -1, 0), wavelengths=wl, names=["BD2165"], verbose=False)
    assert np.allclose(r2["params"]["BD2165"], r["params"]["BD2165"], equal_nan=True)
    da = xr.DataArray(cube, dims=("y", "x", "wavelength"), coords={"wavelength": wl})
    r3 = compute(da, names=["BD2165"], verbose=False)
    assert np.allclose(r3["params"]["BD2165"], r["params"]["BD2165"], equal_nan=True)


def test_image_envi_geotiff_outputs_browse():
    import rasterio
    from rasterio.transform import from_origin
    import spectral.io.envi as envi
    wl = SENSORS["AVIRIS-NG"]
    cube, truth = make_cube(wl)
    tr = from_origin(500000, 4300000, 5, 5)
    # ENVI with map info + reflectance scale factor + bbl (water bands bad)
    bbl = ((wl < 1340) | (wl > 1450)) & ((wl < 1800) | (wl > 1960))
    md = {"wavelength": list(wl), "wavelength units": "Nanometers", "fwhm": [5.5] * wl.size,
          "reflectance scale factor": 10000, "bbl": list(bbl.astype(int)),
          "map info": ["UTM", "1", "1", "500000", "4300000", "5", "5", "11", "North", "WGS-84",
                       "units=Meters"],
          "sensor type": "AVIRIS-NG"}
    envi.save_image(str(TMP / "ang.hdr"), cube, metadata=md, force=True, interleave="bil")
    out = TMP / "ang_crism.tif"
    r = compute(TMP / "ang.hdr", out=out, browse=["PHY", "PAL", "MAF", "FAL", "TRU"],
                block_lines=7, verbose=False)
    p = read_params(out)
    _check_image(p, truth)
    with rasterio.open(out) as ds:
        assert ds.crs is not None and abs(ds.transform.a - 5) < 1e-9
    assert Path(r["browse"]["PHY"]["png"]).with_suffix(".pgw").exists()
    # GeoTIFF with per-band wavelength tags, rasterio band order
    gt = TMP / "cube.tif"
    with rasterio.open(gt, "w", driver="GTiff", height=cube.shape[0], width=cube.shape[1],
                       count=wl.size, dtype="int16", crs="EPSG:32611", transform=tr) as ds:
        ds.write(np.moveaxis(cube, -1, 0))
        for i, w in enumerate(wl, 1):
            ds.update_tags(i, wavelength=f"{w:.2f}", wavelength_units="nm")
    for fmt in ("tif", "img", "nc", "npz"):
        o = TMP / f"gt_crism.{fmt}"
        compute(gt, out=o, names="terrestrial", verbose=False)
        _check_image(read_params(o), truth) if fmt != "x" else None


def test_emit_like_netcdf_swath_and_glt():
    import netCDF4
    wl = SENSORS["EMIT"]
    cube, truth = make_cube(wl, dn=False)
    nl, ns, nb = cube.shape
    p = TMP / "EMIT_L2A_RFL_001_20240101T000000_0000000_000.nc"
    with netCDF4.Dataset(p, "w") as ds:
        ds.createDimension("downtrack", nl); ds.createDimension("crosstrack", ns)
        ds.createDimension("bands", nb)
        v = ds.createVariable("reflectance", "f4", ("downtrack", "crosstrack", "bands"),
                              fill_value=-9999.0)
        v[:] = cube.astype(np.float32)
        g = ds.createGroup("sensor_band_parameters")
        g.createVariable("wavelengths", "f4", ("bands",))[:] = wl
        g.createVariable("fwhm", "f4", ("bands",))[:] = 8.5
        good = ((wl < 1340) | (wl > 1450)) & ((wl < 1800) | (wl > 1960))
        g.createVariable("good_wavelengths", "f4", ("bands",))[:] = good
        loc = ds.createGroup("location")
        loc.createDimension("ortho_y", nl + 4); loc.createDimension("ortho_x", ns + 4)
        gx = np.zeros((nl + 4, ns + 4), np.int32); gy = np.zeros_like(gx)
        yy, xx = np.mgrid[0:nl, 0:ns]
        gx[2:-2, 2:-2] = xx + 1; gy[2:-2, 2:-2] = yy + 1        # shifted by 2 px, 1-based
        loc.createVariable("glt_x", "i4", ("ortho_y", "ortho_x"))[:] = gx
        loc.createVariable("glt_y", "i4", ("ortho_y", "ortho_x"))[:] = gy
        ds.setncattr("geotransform", [-110.0, 0.0005, 0.0, 33.0, 0.0, -0.0005])
    r = compute(p, verbose=False, block_lines=9)
    _check_image(r["params"], truth)
    ro = compute(p, ortho=True, names=["BD2165"], verbose=False, block_lines=9)
    o = ro["params"]["BD2165"]
    assert o.shape == (nl + 4, ns + 4) and np.isnan(o[0, 0])
    assert np.allclose(o[2:-2, 2:-2], r["params"]["BD2165"], equal_nan=True)


def test_prisma_like_he5():
    import h5py
    wl = SENSORS["PRISMA"]
    cube, truth = make_cube(wl, dn=False)
    wv, ws = wl[wl < 1000], wl[wl >= 940]
    vn = cube[..., wl < 1000]; sw = cube[..., wl >= 940]
    lo, hi = 0.0, 1.0
    p = TMP / "PRS_L2D_STD_test.he5"
    with h5py.File(p, "w") as f:
        d = f.create_group("HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields")
        # stored descending wavelength, (lines, bands, samples), uint16 scaled
        d["VNIR_Cube"] = np.moveaxis(np.round(vn[..., ::-1] / hi * 65535), -1, 1).astype(np.uint16)
        d["SWIR_Cube"] = np.moveaxis(np.round(sw[..., ::-1] / hi * 65535), -1, 1).astype(np.uint16)
        f.attrs["List_Cw_Vnir"] = wv[::-1]; f.attrs["List_Cw_Swir"] = ws[::-1]
        f.attrs["L2ScaleVnirMin"] = lo; f.attrs["L2ScaleVnirMax"] = hi
        f.attrs["L2ScaleSwirMin"] = lo; f.attrs["L2ScaleSwirMax"] = hi
        f.attrs["Epsg_Code"] = 32633
        f.attrs["Product_ULcorner_easting"] = 400000.0
        f.attrs["Product_ULcorner_northing"] = 5000000.0
    r = compute(p, verbose=False)
    _check_image(r["params"], truth)
    assert r["crs"].to_epsg() == 32633


def test_enmap_like_folder():
    import rasterio
    from rasterio.transform import from_origin
    wl = SENSORS["EnMAP"]
    cube, truth = make_cube(wl)
    d = TMP / "ENMAP01-____L2A-DT0000001234_20240101T000000Z_001_V010400_20240102T000000Z"
    d.mkdir(exist_ok=True)
    tif = d / (d.name + "-SPECTRAL_IMAGE.TIF")
    with rasterio.open(tif, "w", driver="GTiff", height=cube.shape[0], width=cube.shape[1],
                       count=wl.size, dtype="int16", crs="EPSG:32632",
                       transform=from_origin(600000, 5200000, 30, 30)) as ds:
        ds.write(np.moveaxis(cube, -1, 0))
    bands = "".join(f'<bandID number="{i+1}"><wavelengthCenterOfBand>{w:.3f}</wavelengthCenterOfBand>'
                    f"<FWHMOfBand>8.0</FWHMOfBand><GainOfBand>0.0001</GainOfBand>"
                    f"<OffsetOfBand>0</OffsetOfBand></bandID>" for i, w in enumerate(wl))
    (d / (d.name + "-METADATA.XML")).write_text(f"<level_X><specific><bandCharacterisation>{bands}"
                                                "</bandCharacterisation></specific></level_X>")
    r = compute(d, verbose=False)
    _check_image(r["params"], truth)


def test_fig9_and_matrix():
    from hsi.visualization.crism_sensitivity_matrix import (fig9, write_fig9_template,
                                                            panels_from_dataframe, plot_matrix)
    fig9(out=TMP / "fig9_demo.png", demo=True, csv=TMP / "fig9.csv", dpi=120)
    assert (TMP / "fig9_demo.png").exists()
    # manifest using an Excel file + a USGS pair + text
    tpl = TMP / "man.json"
    write_fig9_template(tpl)
    m = json.loads(tpl.read_text())
    m["panels"][3]["lab"][0].update(file=str(TMP / "spectra.xlsx"), sheet="wide", select="Kaolinite")
    m["panels"][3]["mica"][0].update(file=str(TMP / "kao.sed"))
    tpl.write_text(json.dumps(m))
    fig9(tpl, out=TMP / "fig9_manifest.png", csv=TMP / "fig9_m.csv", dpi=120)
    v = pd.read_csv(TMP / "fig9_m.csv").set_index("label")
    assert v.loc["Kaolinite LAKA04", "BD2165"] > 0.05
    # generic matrix from a compute() table
    wl = SENSORS["EMIT"]
    df = compute(_library(wl), wavelengths=wl, labels=PATCHES, names="fig9")
    df["group"] = ["Phyllosilicates", "Phyllosilicates", "Sulfates", "Carbonates", "Carbonates",
                   "Mafic", "Mafic", "Silica"]
    plot_matrix(panels_from_dataframe(df), TMP / "matrix.png", ncols=2, dpi=120)


def test_pipeline_cli():
    from hsi.pipelines.crism_indices_pipeline import run
    res = run(TMP / "ang.hdr", TMP / "pipe", browse=["PHY", "MAF"])
    assert Path(res["path"]).exists() and (TMP / "pipe" / "ang_coverage.csv").exists()
    df = run(TMP / "spectra.xlsx", TMP / "pipe", matrix=True)
    assert (TMP / "pipe" / "spectra_matrix.png").exists()


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    fails = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            fails += 1
            import traceback
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - fails}/{len(tests)} passed; outputs in {TMP}")
    sys.exit(1 if fails else 0)
