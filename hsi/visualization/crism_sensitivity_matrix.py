"""
crism_sensitivity_matrix.py — summary-product sensitivity matrices in the
style of Viviano-Beck et al. (2014) Figure 9, for ANY spectra.

  plot_matrix(panels, out)                    low-level renderer
  panels_from_dataframe(df, ...)              any compute() table -> panels
  fig9(manifest=..., out=...)                 exact Fig. 9 layout from files
  write_fig9_template(path)                   manifest skeleton

A panel is a dict: {"key": "A", "title": "Ices", "browse": ["IC2"],
                    "blocks": [(name, [row labels], {param: 1-D array}), ...]}
Blocks are stacked with a gap (Fig. 9: lab spectra block, then MICA block).
Cells: linear stretch, 0 (black, band depth <= 0) to the max response
(white) within the panel ("panel") or over all panels ("global").
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from hsi.spectral_indices.crism.parameters import FIG9_COLUMNS
from hsi.spectral_indices.crism.browse import BROWSE

FIG9_LAYOUT = [
    ("A", "Ices", ["IC2"],
     ["H2O Ice GDS136 (77 K)", "CO2 Ice (2mm)"], ["H2O Ice", "CO2 Ice"]),
    ("B", "Fe-oxides and Mafic Silicates", ["MAF"],
     ["Forsterite LASC02", "Fayalite LAPO05", "Plagioclase NCLS04",
      "Orthopyroxene LAPP47B", "Pyroxene NBPP22"],
     ["Mg-Olivine", "Fe-Olivine", "Plagioclase", "Low-Ca Pyroxene", "High-Ca Pyroxene"]),
    ("C", "Sulfates", ["HYD"],
     ["Kieserite F1CC15", "Alunite F1CC08B", "Fe-Hydroxy-Sulfate", "Jarosite LASF21A",
      "Mg Sulfate 799F366", "Gypsum LASF41A", "Bassanite GDS145"],
     ["Kieserite", "Alunite", "Fe-Hydroxy-Sulfate", "Jarosite", "Polyhydrated Sulfate",
      "Gypsum", "Bassanite"]),
    ("D", "Phyllosilicates", ["PHY", "PAL", "PFM"],
     ["Kaolinite LAKA04", "Montmorillonite LAMO02", "Margarite GDS106", "Illite LAIL02",
      "Nontronite NCJB26", "Saponite LASA51", "Talc GDS23", "Serpentine LALZ01",
      "Chlorite LACL14", "Prehnite LAZE03"],
     ["Kaolinite", "Al-Smectite", "Margarite", "Illite", "Fe-Smectite", "Mg-Smectite",
      "Talc", "Serpentine", "Chlorite", "Prehnite"]),
    ("E", "Carbonates", ["CR2"],
     ["Magnesite F1CC06B", "Calcite CAGR01"], ["Mg-Carbonate", "Ca-Carbonate"]),
    ("F", "Halides and Other Hydrated Silicates", ["HYS", "PHY"],
     ["Siliceous Sinter BKR1JB329C", "Epidote GDS26", "Analcime GDS1", "Halide HS433"],
     ["Hydrated Silica", "Epidote", "Analcime", "Chloride"]),
]


# ---------------------------------------------------------------------------
# stretch
# ---------------------------------------------------------------------------
def _needed(columns, panels):
    return set(columns) | {c for p in panels for b in p["browse"] for c in BROWSE[b]}


def stretch_limits(panels, columns, mode="panel"):
    names = _needed(columns, panels)

    def mx(blocks, n):
        vals = [np.ravel(b[2][n]) for b in blocks if n in b[2]]
        if not vals:
            return np.nan
        v = np.concatenate(vals)
        v = v[np.isfinite(v)]
        return float(v.max()) if v.size and v.max() > 0 else np.nan

    if mode == "global":
        allb = [b for p in panels for b in p["blocks"]]
        g = {n: mx(allb, n) for n in names}
        return {p["key"]: g for p in panels}
    return {p["key"]: {n: mx(p["blocks"], n) for n in names} for p in panels}


def _scale(v, hi):
    v = np.asarray(v, float)
    with np.errstate(all="ignore"):
        return np.clip(v / hi, 0, 1) if np.isfinite(hi) else np.zeros_like(v)


# ---------------------------------------------------------------------------
# renderer
# ---------------------------------------------------------------------------
def plot_matrix(panels, out, columns=FIG9_COLUMNS, stretch="panel", ncols=2,
                dpi=300, cell=0.115, label_w=1.55):
    limits = stretch_limits(panels, columns, stretch)
    nc = len(columns)
    gap_rows, head, title_h = 0.6, 1.05, 0.32
    # split panels into columns (balanced by row count)
    colsets = [[] for _ in range(ncols)]
    load = [0] * ncols
    for p in panels:
        n = sum(len(b[1]) for b in p["blocks"])
        j = int(np.argmin(load)) if ncols > 1 else 0
        if ncols == 2 and p["key"] in "ABC":
            j = 0
        elif ncols == 2 and p["key"] in "DEF":
            j = 1
        colsets[j].append(p)
        load[j] += n

    def panel_h(p, top):
        n = sum(len(b[1]) for b in p["blocks"])
        ngap = max(0, len(p["blocks"]) - 1)
        return title_h + (head if top else 0.12) + (n + ngap * gap_rows) * cell + 0.18

    colH = [sum(panel_h(p, i == 0) for i, p in enumerate(cs)) for cs in colsets]
    nb_max = max(len(p["browse"]) for p in panels) if panels else 0
    pw = label_w + nc * cell + 0.15 + nb_max * 0.17 + 0.25
    W, H = ncols * pw + 0.2, max(colH) + 0.1
    fig = plt.figure(figsize=(W, H))
    cmap = ListedColormap(plt.cm.gray(np.linspace(0, 1, 256)))
    cmap.set_bad("white")

    for ci, cs in enumerate(colsets):
        x0 = 0.1 + ci * (pw + 0.1)
        y = H - 0.05
        for i, p in enumerate(cs):
            top, lim, ph = i == 0, limits[p["key"]], panel_h(p, i == 0)
            mats, labels, heights, rgbs = [], [], [], []
            for bi, (bname, lbls, vals) in enumerate(p["blocks"]):
                if bi:
                    mats.append(np.full((1, nc), np.nan)); labels.append(""); heights.append(gap_rows)
                    rgbs.append(np.ones((1, len(p["browse"]), 3)))
                raw = np.column_stack([np.asarray(vals.get(c, np.full(len(lbls), np.nan)), float)
                                       for c in columns])
                m = np.column_stack([_scale(raw[:, j], lim[columns[j]]) for j in range(nc)])
                m[np.all(~np.isfinite(raw), axis=1)] = np.nan
                mats.append(m); labels += list(lbls); heights += [1.0] * len(lbls)
                rgb = np.zeros((len(lbls), len(p["browse"]), 3))
                for k, b in enumerate(p["browse"]):
                    for ch, n in enumerate(BROWSE[b]):
                        if n in vals:
                            rgb[:, k, ch] = np.nan_to_num(_scale(vals[n], lim[n]))
                rgbs.append(rgb)
            mat = np.vstack(mats)
            rgb = np.vstack(rgbs)
            heights = np.array(heights)
            edges = np.r_[0, np.cumsum(heights)]

            ty = y - title_h * 0.75
            fig.text(x0 / W, ty / H, f"{p['key']}.", fontsize=13, fontweight="bold",
                     family="serif", va="baseline")
            fig.text((x0 + 0.3) / W, ty / H, p["title"], fontsize=9, family="serif",
                     va="baseline")
            fig.add_artist(plt.Line2D([x0 / W, (x0 + label_w + nc * cell) / W],
                                      [(ty - 0.05) / H] * 2, color="k", lw=1.2))
            mtop = y - title_h - (head if top else 0.12)
            mh = edges[-1] * cell
            ax = fig.add_axes([(x0 + label_w) / W, (mtop - mh) / H, nc * cell / W, mh / H])
            ax.pcolormesh(np.arange(nc + 1), edges, np.ma.masked_invalid(mat), cmap=cmap,
                          vmin=0, vmax=1, shading="flat")
            ax.set_ylim(edges[-1], 0); ax.set_xlim(0, nc)
            lab = np.array(labels, dtype=object)
            ax.set_yticks(((edges[:-1] + edges[1:]) / 2)[lab != ""])
            ax.set_yticklabels([l for l in labels if l], fontsize=5.2, family="serif")
            ax.tick_params(length=0, pad=1.5)
            if top:
                ax.xaxis.tick_top()
                ax.set_xticks(np.arange(nc) + 0.5)
                ax.set_xticklabels(columns, rotation=90, fontsize=5.2, family="serif")
            else:
                ax.set_xticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            for k, b in enumerate(p["browse"]):
                bx = x0 + label_w + nc * cell + 0.15 + k * 0.17
                bax = fig.add_axes([bx / W, (mtop - mh) / H, 0.11 / W, mh / H])
                for r in range(len(labels)):
                    if labels[r]:
                        bax.add_patch(plt.Rectangle((0, edges[r]), 1, heights[r], color=rgb[r, k]))
                bax.set_xlim(0, 1); bax.set_ylim(edges[-1], 0); bax.axis("off")
                fig.text((bx + 0.055) / W, (mtop + 0.04) / H, b, rotation=90, fontsize=5.5,
                         family="serif", ha="center", va="bottom")
            y -= ph
    fig.savefig(out, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# from a compute() DataFrame
# ---------------------------------------------------------------------------
def panels_from_dataframe(df, group_col="group", label_col="label", block_col=None,
                          browse=None, titles=None):
    """Build panels from a parameter table (output of compute()).
    group_col -> one panel per group; block_col -> sub-blocks within a panel
    (e.g. 'lab' vs 'image'); browse: dict group -> list of browse products."""
    panels = []
    keys = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    groups = df[group_col].unique() if group_col in df else [None]
    for gi, g in enumerate(groups):
        sub = df if g is None else df[df[group_col] == g]
        blocks = []
        bvals = sub[block_col].unique() if block_col else [None]
        for bv in bvals:
            s2 = sub if bv is None else sub[sub[block_col] == bv]
            vals = {c: s2[c].to_numpy(float) for c in s2.columns
                    if c not in (group_col, label_col, block_col)
                    and pd.api.types.is_numeric_dtype(s2[c])}
            blocks.append((str(bv), s2[label_col].astype(str).tolist(), vals))
        panels.append({"key": keys[gi % 26], "title": (titles or {}).get(g, str(g)),
                       "browse": (browse or {}).get(g, []), "blocks": blocks})
    return panels


# ---------------------------------------------------------------------------
# exact Figure 9
# ---------------------------------------------------------------------------
def write_fig9_template(path):
    tpl = {"_help": ("file: any input open_spectral() accepts (csv/xlsx/sli/USGS 'refl|wave'/"
                     "...); select: label or index of the spectrum within that file; "
                     "sheet/orientation/wl_col optional; resample: true for lab spectra "
                     "(Gaussian to CRISM bands), false for CRISM MICA spectra."),
           "panels": [{"key": k, "title": t, "browse": b,
                       "lab": [{"label": l, "file": "", "select": None, "resample": True} for l in lab],
                       "mica": [{"label": l, "file": "", "select": None, "resample": False} for l in mica]}
                      for k, t, b, lab, mica in FIG9_LAYOUT]}
    Path(path).write_text(json.dumps(tpl, indent=2))
    return path


def _one_spectrum(entry):
    from hsi.sensors.open_any import open_spectral
    opts = {k: entry[k] for k in ("sheet", "orientation", "wl_col", "wl_units") if k in entry}
    src = open_spectral(entry["file"], **opts)
    if src["kind"] != "spectra":
        raise ValueError(f"{entry['file']} is an image; give a spectrum file")
    sel = entry.get("select")
    i = 0
    if sel is not None:
        i = sel if isinstance(sel, int) else list(map(str, src["labels"])).index(str(sel))
    return src["wl"], src["data"][i], src.get("scale")


def fig9(manifest=None, out="crism_fig9.png", demo=False, stretch="panel",
         crism_wl=None, fwhm=11.0, csv=None, dpi=300):
    from hsi.spectral_indices.crism.compute import (compute_array, gaussian_resample,
                                                    CRISM_IR_WL)
    from hsi.sensors.source import auto_scale
    wl_c = np.asarray(crism_wl, float) if crism_wl is not None else CRISM_IR_WL
    man = {}
    if manifest:
        for p in json.loads(Path(manifest).read_text())["panels"]:
            man[p["key"]] = p
    rng = np.random.default_rng(0)
    needed = sorted(set(FIG9_COLUMNS) | {c for _, _, b, _, _ in FIG9_LAYOUT for x in b
                                        for c in BROWSE[x]})
    panels, rows_out = [], []
    for key, title, browse, lab_l, mica_l in FIG9_LAYOUT:
        blocks = []
        for grp, labels in (("lab", lab_l), ("mica", mica_l)):
            entries = {e["label"]: e for e in man.get(key, {}).get(grp, [])}
            specs = []
            for lab in labels:
                e = entries.get(lab)
                if demo:
                    from hsi.tests.synthetic_spectra import demo_spectrum
                    specs.append(demo_spectrum(lab, wl_c, rng))
                elif e and e.get("file"):
                    wl, v, sc = _one_spectrum(e)
                    v = v * (sc if sc else auto_scale(v, warn=False))
                    if e.get("resample", grp == "lab"):
                        specs.append(gaussian_resample(v, wl, wl_c, fwhm))
                    else:
                        specs.append(np.interp(wl_c, wl, v, left=np.nan, right=np.nan))
                else:
                    specs.append(np.full(wl_c.shape, np.nan))
            vals = compute_array(np.vstack(specs), wl_c, needed)
            blocks.append((grp, labels, vals))
            for i, lab in enumerate(labels):
                rows_out.append({"panel": key, "group": grp, "label": lab,
                                 **{n: vals[n][i] for n in needed}})
        panels.append({"key": key, "title": title, "browse": browse, "blocks": blocks})
    plot_matrix(panels, out, FIG9_COLUMNS, stretch, dpi=dpi)
    if csv:
        pd.DataFrame(rows_out).to_csv(csv, index=False)
    return out
