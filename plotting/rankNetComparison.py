import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# markers to distinguish different runs inside the same group
MARKERS = ["o", "s", "D", "^", "v", "P", "X"]


# -------------------------------------------------------------------
# CSV utilities
# -------------------------------------------------------------------
def read_csvs_from_folder(folder):
    """Return list of (name, df) for all .csv files in folder, sorted by name."""
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
    files.sort()
    out = []
    for f in files:
        path = os.path.join(folder, f)
        df = pd.read_csv(path)
        df = df.rename(columns=lambda x: x.strip())
        out.append((f, df))
    return out


def detect_size_column(df):
    """Try to find which column to use for bubble size."""
    candidates = ["Estimated Flash Memory (KB)", "TFlite size(KB)"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


# -------------------------------------------------------------------
# Hypervolume helpers  (min RAM, max Accuracy)
# -------------------------------------------------------------------
def collect_ram_acc_flash(
    data_list,
    ram_col="Model RAM (KB)",
    acc_col="Best Test Accuracy",
    flash_col="Estimated Flash Memory (KB)",
) -> np.ndarray:
    """
    From [(fname, df), ...] make a (N,3) array of (RAM, ACC, FLASH).
    """
    pts = []
    for fname, df in data_list:
        for col in (ram_col, acc_col, flash_col):
            if col not in df.columns:
                raise ValueError(f"{fname} missing column '{col}'")

        ram = df[ram_col].astype(float).to_numpy()
        acc = df[acc_col].astype(float).to_numpy()
        flash = df[flash_col].astype(float).to_numpy()
        pts.append(np.column_stack([ram, acc, flash]))

    if not pts:
        return None
    return np.vstack(pts)

def pareto_front_3obj(points: np.ndarray) -> np.ndarray:
    """
    Compute global Pareto front using 3 objectives:
      - col 0: RAM (minimize)
      - col 1: Accuracy (maximize)
      - col 2: Flash (minimize)

    Returns the subset of points that are not dominated in this 3D space.
    """
    pts = points.copy()
    n = pts.shape[0]
    keep = np.ones(n, dtype=bool)
    eps = 1e-12

    for i in range(n):
        if not keep[i]:
            continue

        # For a point j to dominate i:
        # 1) j is no worse in all objectives
        better_eq = (
            (pts[:, 0] <= pts[i, 0] + eps) &   # RAM <=
            (pts[:, 1] >= pts[i, 1] - eps) &   # ACC >=
            (pts[:, 2] <= pts[i, 2] + eps)     # FLASH <=
        )

        # 2) j is strictly better in at least one objective
        strictly_better = (
            (pts[:, 0] < pts[i, 0] - eps) |    # RAM <
            (pts[:, 1] > pts[i, 1] + eps) |    # ACC >
            (pts[:, 2] < pts[i, 2] - eps)      # FLASH <
        )

        dominated = np.any(better_eq & strictly_better)
        if dominated:
            keep[i] = False

    return pts[keep]

def normalize_joint(a: np.ndarray, b: np.ndarray):
    """
    Collect all the values of Ram and Test Accuracy of the models ( Whether they come from With or Without Ranknet )
    Normalize each model to a tuple ( Norm_RAM, Norm_Test_Accuracy ), with each  Norm_Ram,Norm_Test_Accuracy > 0 and < 1
    """
    
    all_pts = np.vstack([a, b])
    ram_min, ram_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    acc_min, acc_max = all_pts[:, 1].min(), all_pts[:, 1].max()

    def norm(p):
        ram_n = (p[:, 0] - ram_min) / (ram_max - ram_min + 1e-12)
        acc_n = (p[:, 1] - acc_min) / (acc_max - acc_min + 1e-12)
        return np.column_stack([ram_n, acc_n])

    return norm(a), norm(b)

def to_minimization(points_norm: np.ndarray) -> np.ndarray:
    """
    To have a good model, we must have as high of an accuracy and as low of Ram
    So, in order to have 2 values which we want to diminish to have a better model:
    We convert to ( Norm_Ram, Norm_Miss_Predictions ), where the ( 0, 0 ) => Best possible model
    
    """
    ram_cost = points_norm[:, 0]
    predic_miss = 1.0 - points_norm[:, 1] # Convert to Norm_Miss_Predictions
    return np.column_stack([ram_cost, predic_miss])

def hypervolume_2d_min(cost_pts: np.ndarray, ref=(1.0, 1.0)) -> float:
    """
    Hypervolume for 2D minimization w.r.t. reference point.

    Sort the Models, on the first places or the array, we will have the models that are consuming the less RamMemory
    So, tuples with the first element to be as small as possible
    """

    if cost_pts.size == 0:
        return 0.0

    """
    Ref = ( 1.0, 1.0 ) is the worst possible model. Talkes about a model that Consumes the most RAM memory,

    Each Model is covering a Rectange with edges:   1) ( ram, miss_accuracy )
                                                    2) ( ram, 1.0 )
                                                    3) ( 1.0, miss_accuracy )
                                                    4) ( 1.0, 1.0 )
    """
    Worst_Norm_RAM, Worst_Norm_Miss_Predictions = ref
    pts = np.asarray(cost_pts, dtype=float)

    # Keep only points that are not worse than the reference
    mask = (pts[:, 0] <= Worst_Norm_RAM) & (pts[:, 1] <= Worst_Norm_Miss_Predictions)
    pts = pts[mask]
    if pts.size == 0:
        return 0.0

    # All x-coordinates where the shape can change: each point's RAM and the ref RAM
    xs = np.unique(np.concatenate([pts[:, 0], [Worst_Norm_RAM]]))
    xs.sort()

    hv = 0.0

    # Sweep over x-slabs [xs[k], xs[k+1]]
    for k in range(len(xs) - 1):
        x_left, x_right = xs[k], xs[k + 1]
        dx = x_right - x_left
        if dx <= 0:
            continue

        # Rectangles that fully cover this slab horizontally: r_i <= x_left
        slab_mask = pts[:, 0] <= x_left
        if not np.any(slab_mask):
            continue

        # The y-intervals [miss_i, R2] of those rectangles
        intervals = np.column_stack([pts[slab_mask, 1], np.full(np.sum(slab_mask), Worst_Norm_Miss_Predictions)])

        # Sort by lower bound and merge to get union length in y
        intervals = intervals[intervals[:, 0].argsort()]
        y_union = 0.0
        cur_lo, cur_hi = None, None

        for lo, hi in intervals:
            if cur_lo is None:
                cur_lo, cur_hi = lo, hi
            elif lo <= cur_hi:  # overlap
                if hi > cur_hi:
                    cur_hi = hi
            else:
                y_union += cur_hi - cur_lo
                cur_lo, cur_hi = lo, hi

        if cur_lo is not None:
            y_union += cur_hi - cur_lo

        hv += dx * y_union

    return hv

# -------------------------------------------------------------------
# Plot + hypervolume
# -------------------------------------------------------------------
def plot_hour_run(root_dir, title=None, marker_scale=1.0, out_path=None):
    """
    root_dir/
      WithRankNet/*.csv
      WithoutRankNet/*.csv
    """
    with_dir = os.path.join(root_dir, "WithRankNet")
    without_dir = os.path.join(root_dir, "WithoutRankNet")

    color_with = "tab:blue"
    color_without = "tab:orange"

    with_RankNet_data = read_csvs_from_folder(with_dir)
    without_RankNet_data = read_csvs_from_folder(without_dir)

    if not with_RankNet_data and not without_RankNet_data:
        raise SystemExit(f"No CSVs found in {with_dir} or {without_dir}")

    # ---------------------------------------------------------
    # 🧮 Hypervolume computation
    # ---------------------------------------------------------
    if with_RankNet_data and without_RankNet_data:
        pts_with_RankNet = collect_ram_acc_flash(with_RankNet_data)
        pts_without_RankNet = collect_ram_acc_flash(without_RankNet_data)

        #Collect the Global Pareto Front from all the runs
        pts_with_pf = pareto_front_3obj(pts_with_RankNet)
        pts_without_pf = pareto_front_3obj(pts_without_RankNet)

        # Collect all the Data and normalize them
        pts_with_2d = pts_with_pf[:, :2]       # RAM, ACC
        pts_without_2d = pts_without_pf[:, :2]
        norm_with, norm_without = normalize_joint(pts_with_2d, pts_without_2d) 

        # Convert the Accuracy to Miss_Prediction
        cost_with = to_minimization(norm_with)
        cost_without = to_minimization(norm_without)

        hv_with = hypervolume_2d_min(cost_with)
        hv_without = hypervolume_2d_min(cost_without)
        improvement = (hv_with - hv_without) / (hv_without + 1e-12) * 100.0

        print(f"\n📁 Evaluating folder: {root_dir}")
        print("📊 Hypervolume Comparison:")
        print(f"   WITHOUT RankNet : {hv_without:.4f}")
        print(f"   WITH RankNet    : {hv_with:.4f}")
        print(f"   Improvement     : {improvement:.2f}%\n")
    else:
        hv_with = hv_without = improvement = None

    # ---------------------------------------------------------
    # 🧭 Title / paths
    # ---------------------------------------------------------
    folder_name = os.path.basename(os.path.normpath(root_dir))
    parent_dir = os.path.basename(os.path.dirname(os.path.normpath(root_dir)))

    constraint_label = ""
    if "unconstrained" in parent_dir.lower():
        constraint_label = "UnConstrained"
    elif "constrained" in parent_dir.lower():
        constraint_label = "Constrained"

    if title is None:
        title = f"{folder_name} {constraint_label}".strip()

    if out_path is None:
        out_path = os.path.join(root_dir, f"{title.replace(' ', '_')}_pareto.png")

    # ---------------------------------------------------------
    # 1) collect ALL size values (both folders) to normalize
    # ---------------------------------------------------------
    all_sizes = []
    size_info = []  # (group, fname, df, size_col)

    for (group_name, data_list) in (("with", with_RankNet_data), ("without", without_RankNet_data)):
        for fname, df in data_list:
            for col in ["Best Test Accuracy", "Model RAM (KB)"]:
                if col not in df.columns:
                    raise ValueError(f"{fname} is missing column '{col}'")

            size_col = detect_size_column(df)
            if size_col is None:
                raise ValueError(f"{fname} has no size column (TFlite/Flash).")

            sizes = df[size_col].to_numpy()
            all_sizes.extend(list(sizes))
            size_info.append((group_name, fname, df, size_col))

    all_sizes = np.array(all_sizes, dtype=float)
    min_size_display, max_size_display = 80, 800

    s_min, s_max = all_sizes.min(), all_sizes.max()

    def normalize_sizes(raw):
        if s_max == s_min:
            return np.full_like(raw, (min_size_display + max_size_display) / 2.0)
        norm = (raw - s_min) / (s_max - s_min)
        return norm * (max_size_display - min_size_display) + min_size_display

    # ---------------------------------------------------------
    # 2) plotting
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    legend_entries = []

    # WITHOUT RankNet first
    idx_without = 0
    for group_name, fname, df, size_col in size_info:
        if group_name != "without":
            continue
        marker = MARKERS[idx_without % len(MARKERS)]
        idx_without += 1

        bubble_sizes = normalize_sizes(df[size_col].to_numpy()) * marker_scale

        plt.scatter(
            df["Model RAM (KB)"],
            df["Best Test Accuracy"],
            s=bubble_sizes,
            alpha=0.7,
            color=color_without,
            marker=marker,
            edgecolors="black",
            zorder=2,
        )
        legend_entries.append(
            (f"No RankNet - {os.path.splitext(fname)[0]}", color_without, marker)
        )

    # WITH RankNet
    idx_with = 0
    for group_name, fname, df, size_col in size_info:
        if group_name != "with":
            continue
        marker = MARKERS[idx_with % len(MARKERS)]
        idx_with += 1

        bubble_sizes = normalize_sizes(df[size_col].to_numpy()) * marker_scale

        plt.scatter(
            df["Model RAM (KB)"],
            df["Best Test Accuracy"],
            s=bubble_sizes,
            alpha=0.9,
            color=color_with,
            marker=marker,
            edgecolors="black",
            zorder=3,
        )
        legend_entries.append(
            (f"With RankNet - {os.path.splitext(fname)[0]}", color_with, marker)
        )

    # labels, grid, invert x
    plt.xlabel("RAM Consumption (KB)", fontsize=12)
    plt.ylabel("Val. Accuracy", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True)
    plt.gca().invert_xaxis()

    # legend
    handles = []
    for text, color, marker in legend_entries:
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                label=text,
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=8,
            )
        )

    plt.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=9,
    )

    # ---------------------------------------------------------
    # 3) annotate hypervolume on the figure (if available)
    # ---------------------------------------------------------
    if hv_with is not None:
        text = (
            f"Hv No RankNet: {hv_without:.3f}\n"
            f"Hv RankNet : {hv_with:.3f}\n"
            f"ΔHv Improvement: {improvement:.1f}%"
        )
        ax = plt.gca()
        ax.text(
            0.02,
            0.02,
            text,
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            verticalalignment="bottom",
        )

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(out_path, dpi=300)
    print(f"✅ saved to {out_path}")
    plt.close()


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot RankNet vs non-RankNet runs from a single hour-run folder."
    )
    parser.add_argument(
        "--hour_run_dir",
        type=str,
        required=True,
        help='Path to the hour-run folder (e.g. "./10Hours"). This folder must contain "WithRankNet" and/or "WithoutRankNet".',
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Title to show on top. If not given, the folder name is used.",
    )
    parser.add_argument(
        "--marker_scale",
        type=float,
        default=1.0,
        help="Extra scale factor for bubble size.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path. If not given, saved inside the hour-run folder.",
    )
    args = parser.parse_args()

    plot_hour_run(
        root_dir=args.hour_run_dir,
        title=args.title,
        marker_scale=args.marker_scale,
        out_path=args.out,
    )
