#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot FEM convergence orders from Scilab-exported CSV files.

Expected CSV format (with header):
Nddl,OL1,OL2,OLinf
625, ...
...

Expected filename pattern (you can override via --pattern):
conv_pb{pb}_P{p}_i{i}_m{m}.csv

Examples:
  python plot_conv.py --p 1 --norm L2
  python plot_conv.py --p 3 --norm Linf 
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


NORM_TO_COL = {
    "L1": 1,     # OL1
    "L2": 2,     # OL2
    "Linf": 3,   # OLinf
}


def read_csv(path: str) -> np.ndarray:
    """
    Reads CSV with a header line and returns a 2D array.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        # single-row CSV -> make it 2D
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError(f"{path}: expected at least 4 columns (Nddl, OL1, OL2, OLinf), got {data.shape[1]}")
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot convergence orders for m=1,2,3 from Scilab CSV outputs.")
    ap.add_argument("--p", type=int, choices=[1, 2, 3], required=True,
                    help="FEM degree: 1->P1, 2->P2, 3->P3")
    ap.add_argument("--norm", choices=["L1", "L2", "Linf"], required=True,
                    help="Which order to plot (column): L1, L2, or Linf")
    args = ap.parse_args()

    col = NORM_TO_COL[args.norm]

    plt.figure()
    any_plotted = False

    for m in (1,2,3):
        fname = f"conv_pb6_P{args.p}_i1_m{m}.csv"
        try:
            data = read_csv(fname)
        except FileNotFoundError:
            print(f"[skip] file not found: {fname}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"[skip] cannot read {fname}: {e}", file=sys.stderr)
            continue

        xvals = data[:, 0]
        yvals = data[:, col]

        plt.plot(xvals, yvals, marker="o", label=f"m={m}")
        any_plotted = True

    if not any_plotted:
        print("No curves plotted (no files found/readable). Check --pb/--i/--pattern and your working directory.",
              file=sys.stderr)
        return 2

    deg_label = f"P{args.p}"
    plt.title(f"Convergence order ({args.norm}) for P{args.p} ")
    plt.xlabel("Nddl")
    plt.ylabel(f"Order ({args.norm})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
