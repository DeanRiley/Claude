#!/usr/bin/env python3
"""
crism_indices_pipeline.py — CRISM summary parameters for any spectral input.

Images (airborne / spaceborne / lab imaging):
    python -m hsi.pipelines.crism_indices_pipeline ang2023..._rfl.hdr -o out/ \
        --names auto --browse PHY PAL MAF HYS CR2
    python -m hsi.pipelines.crism_indices_pipeline EMIT_L2A_RFL_...nc -o out/ --ortho
    -> out/<stem>_crism.tif   (one band per parameter, georeferenced)
       out/<stem>_crism_<BROWSE>.png + .pgw/.prj  (geo-PNGs)
       out/<stem>_coverage.csv  (which parameters the sensor supports)

Spectra (1-D / tables / Excel / libraries):
    python -m hsi.pipelines.crism_indices_pipeline samples.xlsx -o out/ --matrix
    -> out/<stem>_crism.csv  (+ out/<stem>_matrix.png sensitivity matrix)

Figure 9 of Viviano-Beck et al. (2014):
    python -m hsi.pipelines.crism_indices_pipeline --fig9 manifest.json -o out/
    python -m hsi.pipelines.crism_indices_pipeline --fig9-template manifest.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hsi.sensors.open_any import open_spectral                       # noqa: E402
from hsi.spectral_indices.crism.compute import compute, coverage_table  # noqa: E402


def run(inp, outdir, names="auto", browse=None, kernel_mode="nm", scale=None,
        resample_to=None, matrix=False, fmt="tif", block_lines=128, **reader_kw):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    src = open_spectral(inp, **reader_kw)
    stem = Path(str(inp).split("|")[0]).stem.replace(".", "_")
    coverage_table(src["wl"]).to_csv(outdir / f"{stem}_coverage.csv", index=False)
    if src["kind"] == "image":
        out = outdir / f"{stem}_crism.{fmt}"
        res = compute(src, names=names, kernel_mode=kernel_mode, scale=scale,
                      resample_to=resample_to, out=out, browse=browse, browse_dir=outdir,
                      block_lines=block_lines)
        print(f"wrote {out} ({len(res['names'])} parameters)")
        for b, v in (res.get("browse") or {}).items():
            print(f"wrote {v['png']}")
        return res
    df = compute(src, names=names, kernel_mode=kernel_mode, scale=scale,
                 resample_to=resample_to)
    if hasattr(df, "to_frame") and not hasattr(df, "columns"):
        df = df.to_frame().T
    out = outdir / f"{stem}_crism.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} spectra)")
    if matrix:
        from hsi.visualization.crism_sensitivity_matrix import plot_matrix
        from hsi.spectral_indices.crism.parameters import FIG9_COLUMNS
        lab = "label" if "label" in df.columns else df.columns[0]
        cols = [c for c in FIG9_COLUMNS if c in df.columns]
        vals = {c: df[c].to_numpy(float) for c in df.columns if c in set(cols) | set(df.columns[1:])
                and df[c].dtype.kind in "fi"}
        panel = {"key": "A", "title": stem, "browse": [], "blocks": [("", df[lab].astype(str).tolist(), vals)]}
        plot_matrix([panel], outdir / f"{stem}_matrix.png", cols, ncols=1)
        print(f"wrote {outdir / f'{stem}_matrix.png'}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?")
    ap.add_argument("-o", "--outdir", default=".")
    ap.add_argument("--names", nargs="+", default=["auto"],
                    help="auto | terrestrial | fig9 | all | explicit names")
    ap.add_argument("--browse", nargs="*", default=None)
    ap.add_argument("--kernel-mode", choices=["nm", "bands"], default="nm")
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--resample-to", default=None, help="crism | crism_vnir_ir")
    ap.add_argument("--wavelengths", help="text file of band centres if the input lacks them")
    ap.add_argument("--ortho", action="store_true", help="EMIT: apply GLT")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--matrix", action="store_true", help="spectra: draw sensitivity matrix")
    ap.add_argument("--format", default="tif", choices=["tif", "img", "nc", "npz"])
    ap.add_argument("--block-lines", type=int, default=128)
    ap.add_argument("--fig9", help="manifest JSON -> Viviano-Beck Fig. 9")
    ap.add_argument("--fig9-demo", action="store_true")
    ap.add_argument("--fig9-template")
    ap.add_argument("--stretch", choices=["panel", "global"], default="panel")
    a = ap.parse_args()

    if a.fig9_template:
        from hsi.visualization.crism_sensitivity_matrix import write_fig9_template
        print("wrote", write_fig9_template(a.fig9_template))
        return
    if a.fig9 or a.fig9_demo:
        from hsi.visualization.crism_sensitivity_matrix import fig9
        Path(a.outdir).mkdir(parents=True, exist_ok=True)
        out = Path(a.outdir) / "crism_fig9.png"
        fig9(a.fig9, out, demo=a.fig9_demo, stretch=a.stretch,
             csv=Path(a.outdir) / "crism_fig9_values.csv")
        print("wrote", out)
        return
    if not a.input:
        ap.error("input required")
    names = a.names[0] if len(a.names) == 1 and a.names[0] in ("auto", "terrestrial", "fig9", "all") else a.names
    kw = {}
    if a.wavelengths:
        import numpy as np
        from hsi.sensors.tabular_io import read_text_block
        kw["wavelengths"] = read_text_block(a.wavelengths).iloc[:, 0].to_numpy(float)
    if a.ortho:
        kw["ortho"] = True
    if a.sheet:
        kw["sheet"] = a.sheet
    run(a.input, a.outdir, names, a.browse, a.kernel_mode, a.scale, a.resample_to,
        a.matrix, a.format, a.block_lines, **kw)


if __name__ == "__main__":
    main()
