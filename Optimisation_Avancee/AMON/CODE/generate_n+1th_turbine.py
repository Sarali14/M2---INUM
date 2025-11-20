import argparse
import ast
from pathlib import Path
import glob

import numpy as np
import shapefile
from shapely.geometry import Polygon


# --- Config: adapt if needed ---
DATA_DIR = Path("/home/sarah-ali/M2---INUM/Optimisation_Avancee/AMON/data")
SHP_FILE = DATA_DIR / "poly2.shp"


def load_layout_txt(txt_path: Path) -> np.ndarray:
    """
    Load flat [x0, y0, x1, y1, ..., xn, yn] list from a .txt file
    and return an array of shape (n_turbines, 2).
    """
    with open(txt_path, "r") as f:
        content = f.read().strip()

    try:
        flat_list = ast.literal_eval(content)
    except Exception as e:
        raise ValueError(f"Could not parse {txt_path} as a Python list: {e}")

    flat_array = np.asarray(flat_list, dtype=float)

    if flat_array.size % 2 != 0:
        raise ValueError("The file does not contain an even number of values.")

    return flat_array.reshape(-1, 2)


def load_poly2_polygon(shp_file: Path) -> Polygon:
    """Load the largest polygon from poly2.shp."""
    sf = shapefile.Reader(str(shp_file))
    shapes = sf.shapes()
    if len(shapes) == 0:
        raise RuntimeError(f"No shapes found in {shp_file}")

    largest_shape = max(shapes, key=lambda s: Polygon(s.points).area)
    return Polygon(largest_shape.points)


def generate_new_turbine_in_bounds(bounds, existing_points, min_spacing=20.0, max_tries=10000):
    """
    Generate a new turbine inside rectangular bounds:
    - bounds = (min_x, min_y, max_x, max_y)
    - spacing constraint enforced
    """
    min_x, min_y, max_x, max_y = bounds
    existing_points = np.asarray(existing_points)

    for _ in range(max_tries):
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)

        diff = existing_points - np.array([x, y])
        dists = np.hypot(diff[:, 0], diff[:, 1])

        if np.all(dists >= min_spacing):
            return x, y

    raise RuntimeError("Could not find a valid turbine position.")


def next_output_path(base_layout_path: Path, output_dir: Path) -> Path:
    """
    Generate a unique file name inside new_turbines directory.
    """
    stem = base_layout_path.stem  # e.g. Sample_LH_0000
    pattern = str(output_dir / f"{stem}_new_*.txt")
    existing = glob.glob(pattern)

    indices = []
    for path_str in existing:
        name = Path(path_str).stem
        if "_new_" in name:
            try:
                idx = int(name.split("_new_")[-1])
                indices.append(idx)
            except ValueError:
                pass

    next_idx = 0 if not indices else max(indices) + 1
    return output_dir / f"{stem}_new_{next_idx:04d}.txt"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate N new turbine positions inside expanded bounding box "
            "and save EACH one separately inside a directory new_turbines/"
        )
    )
    parser.add_argument("layout_txt", type=str, help="Path to .txt base layout")
    parser.add_argument("--n-new", type=int, default=1, help="Number of new turbines")
    parser.add_argument("--min-spacing", type=float, default=20.0, help="Minimum spacing")
    parser.add_argument("--margin", type=float, default=200.0,
                        help="Margin added around polygon bounding box")

    args = parser.parse_args()
    layout_path = Path(args.layout_txt)

    if not layout_path.exists():
        raise FileNotFoundError(f"{layout_path} does not exist.")

    # ---- Load existing layout ----
    existing_points = load_layout_txt(layout_path)
    all_points = existing_points.copy()

    # ---- Load polygon and compute bounds ----
    poly = load_poly2_polygon(SHP_FILE)
    min_x, min_y, max_x, max_y = poly.bounds
    min_x -= args.margin
    min_y -= args.margin
    max_x += args.margin
    max_y += args.margin

    bounds = (min_x, min_y, max_x, max_y)

    # ---- Prepare output directory ----
    output_dir = layout_path.parent / "new_turbines"
    output_dir.mkdir(exist_ok=True)

    print(f"Saving new turbines in: {output_dir}")

    # ---- Generate new turbines ----
    for k in range(args.n_new):
        print(f"\nGenerating turbine #{k+1} ...")
        new_x, new_y = generate_new_turbine_in_bounds(
            bounds, all_points, min_spacing=args.min_spacing
        )

        print(f"  -> new turbine: [{new_x:.3f}, {new_y:.3f}]")

        # Add to list for spacing consistency
        all_points = np.vstack([all_points, [new_x, new_y]])

        # Save
        out_path = next_output_path(layout_path, output_dir)
        with open(out_path, "w") as f:
            f.write(repr([float(new_x), float(new_y)]))

        print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
