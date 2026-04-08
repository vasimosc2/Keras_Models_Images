import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import re
from pathlib import Path

# markers to distinguish different runs inside the same group
MARKERS = ["o", "s", "D", "^", "v", "P", "X"]


def str_to_bool(value, default=False):
    """
    Convert env/CLI-like strings to bool.
    Accepts: true/false, 1/0, yes/no, on/off
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value

    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def format_improvement(hv_without, hv_with, eps=1e-12):
    """
    Returns a printable improvement string.

    Cases:
      - hv_without == 0 and hv_with == 0  -> 0.00%
      - hv_without == 0 and hv_with > 0   -> ∞
      - otherwise                         -> normal percentage
    """
    if abs(hv_without) <= eps:
        if abs(hv_with) <= eps:
            return "0.00%"
        return "∞"

    improvement = (hv_with - hv_without) / hv_without * 100.0
    return f"{improvement:.2f}%"


def clean_run_label(stem: str) -> str:
    """
    Remove noisy substrings like 'Pareto_optimal_models_dateXXX' from legend labels.
    Works on filename *stem* (no extension).
    """
    s = stem

    # Remove common noisy chunk (case-insensitive) and optional separators
    s = re.sub(r"(?i)pareto[_\- ]*optimal[_\- ]*models[_\- ]*date\d+", "", s)

    # Also remove a shorter variant if it exists
    s = re.sub(r"(?i)pareto[_\- ]*optimal[_\- ]*models", "", s)

    # Cleanup leftover separators/whitespace
    s = re.sub(r"[_\-]+", " ", s).strip()
    s = re.sub(r"\s+", " ", s).strip()

    return s if s else stem


def resolve_vector_outpath(out_path: str | None, root_dir: str, title: str) -> str:
    """
    Ensure output is vector (PDF by default). If user passes .png/.jpg, switch to .pdf.
    """
    if out_path is None:
        return str(Path(root_dir) / f"{title.replace(' ', '_')}_pareto.pdf")

    p = Path(out_path)
    if p.suffix.lower() not in {".pdf", ".svg"}:
        return str(p.with_suffix(".pdf"))
    return str(p)


def normalize_sizes_fixed(
    raw_kb: np.ndarray,
    data_min_kb: float = 0.0,
    data_max_kb: float = 1100.0,
    min_size_display: float = 80.0,
    max_size_display: float = 800.0,
) -> np.ndarray:
    """
    Normalize bubble areas using a FIXED flash range (0..1100 KB) so all figures match.
    Values are clipped to the fixed range.
    """
    raw = np.asarray(raw_kb, dtype=float)
    raw = np.clip(raw, data_min_kb, data_max_kb)

    denom = (data_max_kb - data_min_kb) + 1e-12
    norm = (raw - data_min_kb) / denom
    return norm * (max_size_display - min_size_display) + min_size_display


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
# Hypervolume helpers  (min RAM, max Accuracy, min Flash)
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

        better_eq = (
            (pts[:, 0] <= pts[i, 0] + eps) &
            (pts[:, 1] >= pts[i, 1] - eps) &
            (pts[:, 2] <= pts[i, 2] + eps)
        )

        strictly_better = (
            (pts[:, 0] < pts[i, 0] - eps) |
            (pts[:, 1] > pts[i, 1] + eps) |
            (pts[:, 2] < pts[i, 2] - eps)
        )

        dominated = np.any(better_eq & strictly_better)
        if dominated:
            keep[i] = False

    return pts[keep]


def normalize_joint(a: np.ndarray, b: np.ndarray):
    """
    Collect all values from two sets and normalize to [0,1] jointly.
    """
    all_pts = np.vstack([a, b])
    ram_min, ram_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    acc_min, acc_max = all_pts[:, 1].min(), all_pts[:, 1].max()
    flash_min, flash_max = all_pts[:, 2].min(), all_pts[:, 2].max()

    def norm(p):
        ram_n = (p[:, 0] - ram_min) / (ram_max - ram_min + 1e-12)
        acc_n = (p[:, 1] - acc_min) / (acc_max - acc_min + 1e-12)
        flash_n = (p[:, 2] - flash_min) / (flash_max - flash_min + 1e-12)
        return np.column_stack([ram_n, acc_n, flash_n])

    return norm(a), norm(b)


def to_minimization(points_norm: np.ndarray) -> np.ndarray:
    """
    Convert (Norm_RAM, Norm_ACC, Norm_FLASH) to a minimization problem:
      - RAM cost   = Norm_RAM
      - MISS cost  = 1 - Norm_ACC
      - FLASH cost = Norm_FLASH
    """
    ram_cost = points_norm[:, 0]
    predic_miss = 1.0 - points_norm[:, 1]
    flash_cost = points_norm[:, 2]
    return np.column_stack([ram_cost, predic_miss, flash_cost])


def hypervolume_2d_min(cost_pts: np.ndarray, ref=(1.0, 1.0)) -> float:
    """
    Hypervolume for 2D minimization w.r.t. reference point.
    """
    if cost_pts.size == 0:
        return 0.0

    worst_norm_ram, worst_norm_miss_predictions = ref
    pts = np.asarray(cost_pts, dtype=float)

    mask = (pts[:, 0] <= worst_norm_ram) & (pts[:, 1] <= worst_norm_miss_predictions)
    pts = pts[mask]
    if pts.size == 0:
        return 0.0

    xs = np.unique(np.concatenate([pts[:, 0], [worst_norm_ram]]))
    xs.sort()

    hv = 0.0

    for k in range(len(xs) - 1):
        x_left, x_right = xs[k], xs[k + 1]
        dx = x_right - x_left
        if dx <= 0:
            continue

        slab_mask = pts[:, 0] <= x_left
        if not np.any(slab_mask):
            continue

        intervals = np.column_stack([
            pts[slab_mask, 1],
            np.full(np.sum(slab_mask), worst_norm_miss_predictions)
        ])

        intervals = intervals[intervals[:, 0].argsort()]
        y_union = 0.0
        cur_lo, cur_hi = None, None

        for lo, hi in intervals:
            if cur_lo is None:
                cur_lo, cur_hi = lo, hi
            elif lo <= cur_hi:
                if hi > cur_hi:
                    cur_hi = hi
            else:
                y_union += cur_hi - cur_lo
                cur_lo, cur_hi = lo, hi

        if cur_lo is not None:
            y_union += cur_hi - cur_lo

        hv += dx * y_union

    return hv


def hypervolume_3d_min(cost_pts: np.ndarray, ref=(1.0, 1.0, 1.0)) -> float:
    """
    Hypervolume for 3D minimization w.r.t. reference point.
    """
    if cost_pts.size == 0:
        return 0.0

    pts = np.asarray(cost_pts, dtype=float)
    r1, r2, r3 = ref

    mask = (pts[:, 0] <= r1) & (pts[:, 1] <= r2) & (pts[:, 2] <= r3)
    pts = pts[mask]
    if pts.size == 0:
        return 0.0

    xs = np.unique(np.concatenate([pts[:, 0], [r1]]))
    xs.sort()

    hv = 0.0

    for k in range(len(xs) - 1):
        x_left, x_right = xs[k], xs[k + 1]
        dx = x_right - x_left
        if dx <= 0:
            continue

        slab_mask = pts[:, 0] <= x_left
        if not np.any(slab_mask):
            continue

        yz_pts = pts[slab_mask][:, 1:]
        hv_yz = hypervolume_2d_min(yz_pts, ref=(r2, r3))
        hv += dx * hv_yz

    return hv


def format_flash_size(kb: float) -> str:
    """
    Format a flash size given in KB:
      - < 1024  -> 'XXX KB'
      - >= 1024 -> 'Y.Y MB' (or 'Y MB' if >= 10 MB)
    """
    if kb < 1024:
        return f"{kb:.0f} KB"
    mb = kb / 1024.0
    if mb >= 10:
        return f"{mb:.0f} MB"
    else:
        return f"{mb:.1f} MB"


def compute_combined_hypervolume(with_data, without_data):
    """
    Old behavior:
    combine all WithoutRankNet CSVs into one chunk,
    combine all WithRankNet CSVs into one chunk,
    then compare once.
    """
    pts_with = collect_ram_acc_flash(with_data)
    pts_without = collect_ram_acc_flash(without_data)

    pts_with_pf = pareto_front_3obj(pts_with)
    pts_without_pf = pareto_front_3obj(pts_without)

    norm_with, norm_without = normalize_joint(pts_with_pf, pts_without_pf)
    cost_with = to_minimization(norm_with)
    cost_without = to_minimization(norm_without)

    hv_with = hypervolume_3d_min(cost_with)
    hv_without = hypervolume_3d_min(cost_without)
    improvement = format_improvement(hv_without, hv_with)

    return {
        "hv_without": hv_without,
        "hv_with": hv_with,
        "improvement": improvement,
    }


def compute_all_vs_all_hypervolume(with_data, without_data):
    """
    Current behavior:
    every WithoutRankNet file against every WithRankNet file.
    """
    results = []

    for i, (without_fname, without_df) in enumerate(without_data, start=1):
        for j, (with_fname, with_df) in enumerate(with_data, start=1):
            pts_without = collect_ram_acc_flash([(without_fname, without_df)])
            pts_with = collect_ram_acc_flash([(with_fname, with_df)])

            pts_without_pf = pareto_front_3obj(pts_without)
            pts_with_pf = pareto_front_3obj(pts_with)

            norm_with, norm_without = normalize_joint(pts_with_pf, pts_without_pf)
            cost_with = to_minimization(norm_with)
            cost_without = to_minimization(norm_without)

            hv_with = hypervolume_3d_min(cost_with)
            hv_without = hypervolume_3d_min(cost_without)
            improvement = format_improvement(hv_without, hv_with)

            results.append({
                "without_index": i,
                "with_index": j,
                "without_file": without_fname,
                "with_file": with_fname,
                "hv_without": hv_without,
                "hv_with": hv_with,
                "improvement": improvement,
            })

    return results


def plot_hour_run_new(root_dir, title=None, marker_scale=1.0, out_path=None, combine_runs=False):
    """
    Paper-style layout:
      - Left: plot
      - Right: 3 stacked legend boxes (Flash / Runs / MCU)

    Hypervolume mode:
      - combine_runs=True  -> old combined mode
      - combine_runs=False -> all-vs-all mode
    """
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # ---------------------------------------------------------
    # Load CSVs
    # ---------------------------------------------------------
    with_dir = os.path.join(root_dir, "WithRankNet")
    without_dir = os.path.join(root_dir, "WithoutRankNet")

    color_with = "tab:blue"
    color_without = "tab:orange"

    with_ranknet_data = read_csvs_from_folder(with_dir)
    without_ranknet_data = read_csvs_from_folder(without_dir)

    if not with_ranknet_data and not without_ranknet_data:
        raise SystemExit(f"No CSVs found in {with_dir} or {without_dir}")

    # ---------------------------------------------------------
    # Hypervolume computation
    # ---------------------------------------------------------
    combined_result = None
    all_vs_all_results = []

    if with_ranknet_data and without_ranknet_data:
        print(f"\n📁 Evaluating folder: {root_dir}")
        print(f"⚙️ combine_runs = {combine_runs}\n")

        if combine_runs:
            combined_result = compute_combined_hypervolume(with_ranknet_data, without_ranknet_data)

            print("📊 Combined Hypervolume Comparison:")
            print(f"   HV WITHOUT RankNet : {combined_result['hv_without']:.4f}")
            print(f"   HV WITH RankNet    : {combined_result['hv_with']:.4f}")
            print(f"   Improvement        : {combined_result['improvement']}\n")
        else:
            all_vs_all_results = compute_all_vs_all_hypervolume(with_ranknet_data, without_ranknet_data)

            print("📊 All-vs-All Hypervolume Comparison:")
            for res in all_vs_all_results:
                print(
                    f"   Without Run {res['without_index']} ({res['without_file']}) "
                    f"vs With Run {res['with_index']} ({res['with_file']})"
                )
                print(f"      HV WITHOUT RankNet : {res['hv_without']:.4f}")
                print(f"      HV WITH RankNet    : {res['hv_with']:.4f}")
                print(f"      Improvement        : {res['improvement']}")
            print()
    else:
        print(f"\n📁 Evaluating folder: {root_dir}")
        print("⚠️ Hypervolume comparison skipped because one of the folders is empty.\n")

    # ---------------------------------------------------------
    # Title / constrained label
    # ---------------------------------------------------------
    folder_name = os.path.basename(os.path.normpath(root_dir))
    parent_dir = os.path.basename(os.path.dirname(os.path.normpath(root_dir)))

    constraint_label = ""
    if "unconstrained" in parent_dir.lower():
        constraint_label = "UnConstrained"
    elif "constrained" in parent_dir.lower():
        constraint_label = "Constrained"

    is_constrained = (constraint_label == "Constrained")

    if title is None:
        title = f"{folder_name} {constraint_label}".strip()

    out_path = resolve_vector_outpath(out_path, root_dir, title)

    # ---------------------------------------------------------
    # Gather plot inputs
    # ---------------------------------------------------------
    size_info = []
    for group_name, data_list in (("without", without_ranknet_data), ("with", with_ranknet_data)):
        for fname, df in data_list:
            for col in ["Best Test Accuracy", "Model RAM (KB)"]:
                if col not in df.columns:
                    raise ValueError(f"{fname} is missing column '{col}'")

            size_col = detect_size_column(df)
            if size_col is None:
                raise ValueError(f"{fname} has no size column (TFlite/Flash).")

            size_info.append((group_name, fname, df, size_col))

    # ---------------------------------------------------------
    # 2-column layout: plot + legend-column
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(12.5, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.0, 1.9], wspace=0.06)

    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.axis("off")

    legend_entries = []

    # WITHOUT RankNet first
    run_id_without = 0
    idx_without = 0
    for group_name, fname, df, size_col in size_info:
        if group_name != "without":
            continue

        marker = MARKERS[idx_without % len(MARKERS)]
        idx_without += 1

        bubble_sizes = normalize_sizes_fixed(df[size_col].to_numpy()) * marker_scale

        ax.scatter(
            df["Model RAM (KB)"],
            df["Best Test Accuracy"],
            s=bubble_sizes,
            alpha=0.75,
            color=color_without,
            marker=marker,
            edgecolors="black",
            linewidths=0.7,
            zorder=2,
        )

        run_id_without += 1
        legend_entries.append((f"No RankNet - Run {run_id_without}", color_without, marker))

    # WITH RankNet
    run_id_with = 0
    idx_with = 0
    for group_name, fname, df, size_col in size_info:
        if group_name != "with":
            continue

        marker = MARKERS[idx_with % len(MARKERS)]
        idx_with += 1

        bubble_sizes = normalize_sizes_fixed(df[size_col].to_numpy()) * marker_scale

        ax.scatter(
            df["Model RAM (KB)"],
            df["Best Test Accuracy"],
            s=bubble_sizes,
            alpha=0.9,
            color=color_with,
            marker=marker,
            edgecolors="black",
            linewidths=0.7,
            zorder=3,
        )

        run_id_with += 1
        legend_entries.append((f"With RankNet - Run {run_id_with}", color_with, marker))

    # ---------------------------------------------------------
    # Fixed axes
    # ---------------------------------------------------------
    ax.set_title(title)
    ax.set_xlabel("RAM Consumption (KB)")
    ax.set_ylabel("Val. Accuracy")

    ax.set_ylim(0.3, 0.75)
    ax.set_xlim(1200, 0)
    ax.grid(True, alpha=0.35, linewidth=1.0)

    # ---------------------------------------------------------
    # Legend box 1: Flash Memory Range
    # ---------------------------------------------------------
    fmin, fmax = 0.0, 1100.0
    bands = [
        (fmin, fmin + (fmax - fmin) / 3),
        (fmin + (fmax - fmin) / 3, fmin + 2 * (fmax - fmin) / 3),
        (fmin + 2 * (fmax - fmin) / 3, fmax),
    ]
    sample_vals = np.array([(a + b) / 2 for a, b in bands], dtype=float)
    sample_sizes = normalize_sizes_fixed(sample_vals) * marker_scale
    sample_labels = [
        f"{format_flash_size(bands[0][0])}–{format_flash_size(bands[0][1])}",
        f"{format_flash_size(bands[1][0])}–{format_flash_size(bands[1][1])}",
        f"{format_flash_size(bands[2][0])}–{format_flash_size(bands[2][1])}",
    ]

    bubble_handles = [
        ax_leg.scatter(
            [], [], s=s, color="tab:blue", alpha=0.85,
            edgecolors="black", linewidths=1.2, label=lab
        )
        for s, lab in zip(sample_sizes, sample_labels)
    ]

    bubble_legend = ax_leg.legend(
        handles=bubble_handles,
        title="Flash Memory Range",
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        frameon=True,
        borderpad=1.0,
        labelspacing=1.2,
        handletextpad=1.0,
        title_fontsize=12,
    )
    ax_leg.add_artist(bubble_legend)

    # ---------------------------------------------------------
    # Legend box 2: Runs
    # ---------------------------------------------------------
    run_handles = [
        Line2D(
            [0], [0],
            marker=mk, color="w",
            label=txt,
            markerfacecolor=col,
            markeredgecolor="black",
            markersize=8
        )
        for txt, col, mk in legend_entries
    ]

    run_legend = ax_leg.legend(
        handles=run_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.62),
        frameon=True,
        borderpad=1.0,
        labelspacing=0.6,
        handletextpad=0.8,
    )
    ax_leg.add_artist(run_legend)

    # ---------------------------------------------------------
    # Legend box 3: MCU budgets (only constrained)
    # ---------------------------------------------------------
    if is_constrained:
        mcu_devices = [
            ("Arduino Nano 33 BLE", 256, 1024),
            ("STM32F411 (Nucleo)", 128, 512),
            ("Raspberry Pi Pico", 264, 2048),
            ("ESP32-C3 DevKit", 400, 4096),
        ]

        mcu_handles = []
        for name, ram_kb, flash_kb in mcu_devices:
            label = f"{name}: {ram_kb} KB RAM, {format_flash_size(flash_kb)} Flash"
            mcu_handles.append(
                Line2D(
                    [0], [0],
                    marker="o", linestyle="",
                    color="gray",
                    markerfacecolor="none",
                    markeredgecolor="gray",
                    label=label
                )
            )

        mcu_legend = ax_leg.legend(
            handles=mcu_handles,
            title="Typical MCU Budgets",
            loc="upper left",
            bbox_to_anchor=(-0.06, 0.20),
            frameon=True,
            borderpad=1.0,
            labelspacing=0.6,
            handletextpad=0.8,
            title_fontsize=12,
        )
        ax_leg.add_artist(mcu_legend)

    # ---------------------------------------------------------
    # Hypervolume annotation
    # ---------------------------------------------------------
    if combined_result is not None:
        text = (
            f"Combined:\n"
            f"NoRN={combined_result['hv_without']:.3f}\n"
            f"RN={combined_result['hv_with']:.3f}\n"
            f"Δ={combined_result['improvement']}"
        )

        ax.text(
            0.02, 0.02, text,
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            verticalalignment="bottom",
        )

    elif all_vs_all_results:
        lines = []
        for res in all_vs_all_results:
            lines.append(
                f"Wo{res['without_index']} vs W{res['with_index']}: "
                f"NoRN={res['hv_without']:.3f}, "
                f"RN={res['hv_with']:.3f}, "
                f"Δ={res['improvement']}"
            )

        text = "\n".join(lines)

        ax.text(
            0.02, 0.02, text,
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            verticalalignment="bottom",
        )

    fig.savefig(out_path)
    print(f"✅ saved to {out_path}")
    plt.close(fig)


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
    parser.add_argument(
        "--combine_runs",
        type=str,
        default=None,
        help=(
            "Whether to combine all runs before HV computation. "
            "Accepted values: true/false. "
            "If omitted, environment variable COMBINE_RUNS is used. "
            "If that is also missing, default is false."
        ),
    )
    args = parser.parse_args()

    combine_runs = str_to_bool(
        args.combine_runs if args.combine_runs is not None else os.getenv("COMBINE_RUNS"),
        default=False
    )

    plot_hour_run_new(
        root_dir=args.hour_run_dir,
        title=args.title,
        marker_scale=args.marker_scale,
        out_path=args.out,
        combine_runs=combine_runs,
    )