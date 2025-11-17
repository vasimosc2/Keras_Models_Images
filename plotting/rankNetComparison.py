import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# markers to distinguish different runs inside the same group
MARKERS = ["o", "s", "D", "^", "v", "P", "X"]


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


def plot_hour_run(root_dir, title=None, marker_scale=1.0, out_path=None):
    """
    root_dir/
      WithRankNet/*.csv
      WithoutRankNet/*.csv
    """
    with_dir = os.path.join(root_dir, "WithRankNet")
    without_dir = os.path.join(root_dir, "WithoutRankNet")

    # colors for the 2 categories
    color_with = "tab:blue"
    color_without = "tab:orange"

    with_data = read_csvs_from_folder(with_dir)
    without_data = read_csvs_from_folder(without_dir)

    if not with_data and not without_data:
        raise SystemExit(f"No CSVs found in {with_dir} or {without_dir}")

    # ---------------------------------------------------------
    # 🧭 Detect base folder name and constraint type
    # ---------------------------------------------------------
    folder_name = os.path.basename(os.path.normpath(root_dir))
    parent_dir = os.path.basename(os.path.dirname(os.path.normpath(root_dir)))

    # Determine constraint label from parent directory name
    constraint_label = ""
    if "unconstrained" in parent_dir.lower():
        constraint_label = "UnConstrained"
    elif "constrained" in parent_dir.lower():
        constraint_label = "Constrained"

    # Final title
    if title is None:
        title = f"{folder_name} {constraint_label}".strip()

    if out_path is None:
        out_path = os.path.join(root_dir, f"{title.replace(' ', '_')}_pareto.png")

    # ---------------------------------------------------------
    # 1) collect ALL size values (both folders) to normalize
    # ---------------------------------------------------------
    all_sizes = []
    size_info = []  # (group, fname, df, size_col)

    # store info for later plotting
    for (group_name, data_list) in (("with", with_data), ("without", without_data)):
        for fname, df in data_list:
            # basic checks for x,y
            for col in ["Best Test Accuracy", "Model RAM (KB)"]:
                if col not in df.columns:
                    raise ValueError(f"{fname} is missing column '{col}'")

            size_col = detect_size_column(df)
            if size_col is None:
                raise ValueError(f"{fname} has no size column (TFlite/Flash).")

            sizes = df[size_col].to_numpy()
            all_sizes.extend(list(sizes))
            size_info.append((group_name, fname, df, size_col))

    # normalize sizes to a nice range
    all_sizes = np.array(all_sizes, dtype=float)
    min_size_display, max_size_display = 80, 800  # tweak here
    s_min, s_max = all_sizes.min(), all_sizes.max()
    # avoid division by zero
    def normalize_sizes(raw):
        if s_max == s_min:
            return np.full_like(raw, (min_size_display + max_size_display) / 2.0)
        norm = (raw - s_min) / (s_max - s_min)
        return norm * (max_size_display - min_size_display) + min_size_display

    # ---------------------------------------------------------
    # 2) do the actual plotting
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    legend_entries = []

    # plot WITH RankNet second (so it appears on top)
    # first: WITHOUT
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

    # then: WITH
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
            zorder=3,  # on top
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

    # build legend
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

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(out_path, dpi=300)
    print(f"✅ saved to {out_path}")
    plt.close()


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
        default=1.0,   # 1.0 because we already normalize
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
