# CRISM summary parameters for any spectral data

Viviano-Beck et al. (2014), *Revised CRISM spectral parameters and summary
products based on the currently detected mineral diversity on Mars*,
JGR Planets 119, 1403–1431, doi:10.1002/2014JE004627. Formulas and kernel
widths follow the PDS Geosciences Node *CRISM Summary Parameter Calculation
Reference* (the CAT implementation).

These are plain functions organised by directory, with no classes. Each part
goes in its toolbox component:

```
hsi/
  sensors/                    I/O -> a "source" dict (see sensors/source.py)
    open_any.py               open_spectral(anything) dispatcher
    tabular_io.py             numpy, pandas, CSV/TXT/Excel/ODS/parquet, SED/ASD txt,
                              USGS splib07, ENVI .sli, ASD binary (specdal)
    envi_io.py                ENVI cubes: AVIRIS-C/NG/3, HySpex, AISA, HyMap, SisuROCK…
    geotiff_io.py             GeoTIFF / any GDAL raster (+ wavelength discovery)
    spaceborne_io.py          EMIT L2A (swath or GLT ortho), PRISMA L2B/C/D, EnMAP L2A
    xarray_io.py              xarray DataArray/Dataset, generic netCDF/Zarr
  spectral_indices/crism/
    kernels.py                median-kernel R(λ), continuum, band depth
    parameters.py             60 parameters + PARAMETERS registry, TERRESTRIAL, FIG9_COLUMNS
    browse.py                 browse RGB products, stretch, geo-PNG
    compute.py                compute(anything) / compute_array / coverage_table
    outputs.py                streaming GeoTIFF / ENVI writers, netCDF, npz
  visualization/crism_sensitivity_matrix.py   Fig. 9-style matrices (+ exact Fig. 9)
  pipelines/crism_indices_pipeline.py         CLI: cube -> GeoTIFF + geo-PNGs + coverage CSV
  tests/                      synthetic spectra + end-to-end tests over every format
```

## Python

```python
from hsi.spectral_indices.crism.compute import compute, coverage_table

compute(spec, wavelengths=wl)                 # 1-D -> pandas Series
compute(lib, wavelengths=wl, labels=names)    # 2-D (N,B) -> DataFrame
compute(df)                                   # wide or long DataFrame; ID/assay columns kept
compute("samples.xlsx")                       # every spectral sheet; or sheet="Sheet2"
compute("kaol.txt|splib07a_Wavelengths_ASD.txt")   # USGS splib07
compute(cube, wavelengths=wl)                 # (rows,cols,B) or (B,rows,cols) -> dict of 2-D
compute(xr_dataarray)
compute("ang2023_rfl.hdr", out="ang_crism.tif", browse=["PHY","PAL","MAF"])
compute("EMIT_L2A_RFL_....nc", ortho=True, out="emit_crism.tif")
compute("PRS_L2D_STD_....he5", out="prisma_crism.tif")
compute("ENMAP01-____L2A-.../", out="enmap_crism.tif")
compute(lab_1nm_df, resample_to="crism")      # emulate CRISM bands first

coverage_table(wl)                            # which parameters a sensor supports
```

You can choose which parameters to compute with `names=`:
- `"auto"` (default) computes every parameter the sensor covers.
- `"terrestrial"` computes the VSWIR set that avoids the atmospheric windows.
- `"fig9"` computes the 26 parameters shown in Fig. 9.
- `"all"` computes everything.
- You can also pass an explicit list of names.

## CLI

```
python -m hsi.pipelines.crism_indices_pipeline ang_rfl.hdr -o out/ --browse PHY PAL MAF
python -m hsi.pipelines.crism_indices_pipeline EMIT_L2A_RFL_...nc -o out/ --ortho
python -m hsi.pipelines.crism_indices_pipeline samples.xlsx -o out/ --matrix
python -m hsi.pipelines.crism_indices_pipeline --fig9-template fig9.json
python -m hsi.pipelines.crism_indices_pipeline --fig9 fig9.json -o out/
python hsi/tests/test_crism_indices.py
```

## How non-CRISM data are handled

- **Kernel widths.** By default (`kernel_mode="nm"`), a CRISM kernel of k bands is
  turned into a wavelength window of ±((k//2)·6.55 + 3.3) nm. All bands inside
  that window are median-filtered. On 1 nm lab data this gives smoothing
  equivalent to CRISM's. On coarser sensors (EMIT at 7.4 nm, PRISMA and EnMAP
  at about 10 nm, HyMap at 15 nm) the window shrinks to the nearest band.
  `kernel_mode="bands"` reproduces CAT exactly on CRISM data; the tests check
  that the two modes give identical results on the CRISM grid.
- **Coverage.** A band centre must have a band within max(10 nm, 0.75 × spacing)
  of it. Continuum anchors may substitute the nearest valid band up to
  `anchor_tol_nm` (40 nm) away. This keeps three groups of parameters
  available:
  - D2300, HCPINDEX2 and ISLOPE1, whose 2530 nm anchor lies past the edge of
    2.5 µm sensors;
  - D2200, D2300 and BD1750_2, whose 1815 nm anchor sits inside the H₂O
    bad-band window;
  - BD1500_2.

  `coverage_table` lists every substitution. It decides coverage by running
  each parameter on a flat spectrum, so it always agrees with `compute`.
  Set `anchor_tol_nm=0` for strict CRISM behaviour.
- **Bad bands.** Masks from ENVI `bbl` and EMIT `good_wavelengths` are applied.
  Bands that are NaN in every pixel are dropped.
- **Scaling.** The reflectance scale is read from metadata where available
  (ENVI `reflectance scale factor`, EnMAP gains, PRISMA L2Scale). Otherwise it
  is auto-detected: 0–10000 integers and percent reflectance are both
  converted to 0–1.
- **Large images.** Images are processed in blocks of lines (`block_lines`).
  GeoTIFF and ENVI outputs are streamed to disk, keeping the CRS and transform.
- **Departures from CAT.** BDI1000VIS, BDI1000IR and BDI2000 integrate over µm;
  multiply by about 152.7 to compare with CAT's per-band sums. VAR is scaled
  to the 189 bands CRISM has between 1021 and 2253 nm. Both changes keep the
  values comparable across sensors.

## Caveats

- I did not check the CHL and CAR browse definitions against Table 3 of the
  paper.
- Parameters inside the terrestrial H₂O windows will be NaN or meaningless on
  airborne and spaceborne data: BD1400, BD1435, BD1900_2, BD1900r2,
  OLINDEX3's 1862 nm anchor, and LCPINDEX2's 1870 nm band. That is expected.
- The EMIT, PRISMA and EnMAP readers have been tested only on synthetic files
  built to each product's documented structure. Run one real granule of each
  before relying on them.
