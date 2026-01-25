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


def normalize_sizes_fixed(raw_kb: np.ndarray,
                          data_min_kb: float = 0.0,
                          data_max_kb: float = 1100.0,
                          min_size_display: float = 80.0,
                          max_size_display: float = 800.0) -> np.ndarray:
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
    flash_min, flash_max = all_pts[:, 2].min(), all_pts[:, 2].max()

    def norm(p):
        ram_n = (p[:, 0] - ram_min) / (ram_max - ram_min + 1e-12)
        acc_n = (p[:, 1] - acc_min) / (acc_max - acc_min + 1e-12)
        flash_n = (p[:, 2] - flash_min) / (flash_max - flash_min + 1e-12)
        return np.column_stack([ram_n, acc_n, flash_n])

    return norm(a), norm(b)

def to_minimization(points_norm: np.ndarray) -> np.ndarray:
    """
    To have a good model, we must have as high of an accuracy and as low of Ram
    So, in order to have 2 values which we want to diminish to have a better model:
    We convert to ( Norm_Ram, Norm_Miss_Predictions ), where the ( 0, 0 ) => Best possible model
    
    """
    ram_cost = points_norm[:, 0]
    predic_miss = 1.0 - points_norm[:, 1] # Convert to Norm_Miss_Predictions
    flash_cost = points_norm[:, 2]
    return np.column_stack([ram_cost, predic_miss, flash_cost])

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

def format_flash_size(kb: float) -> str:
    """
    Format a flash size given in KB:
      - < 1024  → 'XXX KB'
      - >= 1024 → 'Y.Y MB' (or 'Y MB' if >= 10 MB)
    """
    if kb < 1024:
        return f"{kb:.0f} KB"
    mb = kb / 1024.0
    if mb >= 10:
        return f"{mb:.0f} MB"
    else:
        return f"{mb:.1f} MB"

def hypervolume_3d_min(cost_pts: np.ndarray, ref=(1.0, 1.0, 1.0)) -> float:
    """
    Hypervolume for 3D minimization w.r.t. reference point.

    cost_pts: shape (N, 3) with (RAM_cost, MISS_cost, FLASH_cost),
              all in [0,1], smaller = better.
    ref: worst point (R1, R2, R3), typically (1,1,1).

    Strategy:
      - Sweep along RAM (x) in slabs.
      - For each x-slab, consider points whose RAM_cost <= x_left.
      - In that slab, compute 2D HV in (MISS, FLASH) using hypervolume_2d_min.
      - Integrate over x to get 3D HV.
    """
    if cost_pts.size == 0:
        return 0.0

    pts = np.asarray(cost_pts, dtype=float)
    R1, R2, R3 = ref

    # Keep only points no worse than reference
    mask = (pts[:, 0] <= R1) & (pts[:, 1] <= R2) & (pts[:, 2] <= R3)
    pts = pts[mask]
    if pts.size == 0:
        return 0.0

    # All x (RAM) where the frontier can change
    xs = np.unique(np.concatenate([pts[:, 0], [R1]]))
    xs.sort()

    hv = 0.0

    for k in range(len(xs) - 1):
        x_left, x_right = xs[k], xs[k + 1]
        dx = x_right - x_left
        if dx <= 0:
            continue

        # Points whose rectangle covers this x_slab: RAM_cost <= x_left
        slab_mask = pts[:, 0] <= x_left
        if not np.any(slab_mask):
            continue

        # Project to 2D (MISS, FLASH) for this slab
        yz_pts = pts[slab_mask][:, 1:]  # shape (M, 2)

        # 2D HV in (MISS, FLASH) with ref=(R2,R3)
        hv_yz = hypervolume_2d_min(yz_pts, ref=(R2, R3))

        hv += dx * hv_yz

    return hv
def plot_hour_run_new(root_dir, title=None, marker_scale=1.0, out_path=None):
    """
    Paper-style layout:
      - Left: plot
      - Right: 3 stacked legend boxes (Flash / Runs / MCU)

    Requirements covered:
      - Vector output (PDF/SVG)
      - Fixed axes: Accuracy 0.3..0.7, RAM 30..150 (inverted)
      - Bubble sizes normalized with FIXED Flash range 0..1100 KB
      - Run legend labels are: "No RankNet - Run 1", "With RankNet - Run 1", etc.
    """
    import re
    from pathlib import Path

    # ---------------- Publication-ish styling ----------------
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "pdf.fonttype": 42,  # embed TrueType fonts in PDF (copy-paste text)
        "ps.fonttype": 42,
    })

    def resolve_vector_outpath(out_path_local: str | None, root: str, ttl: str) -> str:
        if out_path_local is None:
            return str(Path(root) / f"{ttl.replace(' ', '_')}_pareto.pdf")
        p = Path(out_path_local)
        if p.suffix.lower() not in {".pdf", ".svg"}:
            return str(p.with_suffix(".pdf"))
        return str(p)

    def normalize_sizes_fixed(raw_kb: np.ndarray,
                              data_min_kb: float = 0.0,
                              data_max_kb: float = 1100.0,
                              min_size_display: float = 80.0,
                              max_size_display: float = 800.0) -> np.ndarray:
        raw = np.asarray(raw_kb, dtype=float)
        raw = np.clip(raw, data_min_kb, data_max_kb)
        denom = (data_max_kb - data_min_kb) + 1e-12
        norm = (raw - data_min_kb) / denom
        return norm * (max_size_display - min_size_display) + min_size_display

    # ---------------------------------------------------------
    # Load CSVs
    # ---------------------------------------------------------
    with_dir = os.path.join(root_dir, "WithRankNet")
    without_dir = os.path.join(root_dir, "WithoutRankNet")

    color_with = "tab:blue"
    color_without = "tab:orange"

    with_RankNet_data = read_csvs_from_folder(with_dir)
    without_RankNet_data = read_csvs_from_folder(without_dir)

    if not with_RankNet_data and not without_RankNet_data:
        raise SystemExit(f"No CSVs found in {with_dir} or {without_dir}")

    # ---------------------------------------------------------
    # Hypervolume computation
    # ---------------------------------------------------------
    if with_RankNet_data and without_RankNet_data:
        pts_with_RankNet = collect_ram_acc_flash(with_RankNet_data)
        pts_without_RankNet = collect_ram_acc_flash(without_RankNet_data)

        pts_with_pf = pareto_front_3obj(pts_with_RankNet)
        pts_without_pf = pareto_front_3obj(pts_without_RankNet)

        norm_with, norm_without = normalize_joint(pts_with_pf, pts_without_pf)
        cost_with = to_minimization(norm_with)
        cost_without = to_minimization(norm_without)

        hv_with = hypervolume_3d_min(cost_with)
        hv_without = hypervolume_3d_min(cost_without)
        improvement = (hv_with - hv_without) / (hv_without + 1e-12) * 100.0

        print(f"\n📁 Evaluating folder: {root_dir}")
        print("📊 Hypervolume Comparison:")
        print(f"   WITHOUT RankNet : {hv_without:.4f}")
        print(f"   WITH RankNet    : {hv_with:.4f}")
        print(f"   Improvement     : {improvement:.2f}%\n")
    else:
        hv_with = hv_without = improvement = None

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
    size_info = []  # (group, fname, df, size_col)
    for (group_name, data_list) in (("without", without_RankNet_data), ("with", with_RankNet_data)):
        for fname, df in data_list:
            for col in ["Best Test Accuracy", "Model RAM (KB)"]:
                if col not in df.columns:
                    raise ValueError(f"{fname} is missing column '{col}'")

            size_col = detect_size_column(df)
            if size_col is None:
                raise ValueError(f"{fname} has no size column (TFlite/Flash).")

            size_info.append((group_name, fname, df, size_col))

    # ---------------------------------------------------------
    # 2-column layout: plot + legend-column (robust; never cropped)
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(12.5, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.0, 1.9], wspace=0.06)

    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.axis("off")

    legend_entries = []

    # ✅ run counters (separate per group)
    run_id_without = 0
    run_id_with = 0

    # WITHOUT RankNet first
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

        # ✅ label as Run 1 / Run 2 ...
        run_id_without += 1
        legend_entries.append((f"No RankNet - Run {run_id_without}", color_without, marker))

    # WITH RankNet
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

        # ✅ label as Run 1 / Run 2 ...
        run_id_with += 1
        legend_entries.append((f"With RankNet - Run {run_id_with}", color_with, marker))

    # ---------------------------------------------------------
    # Fixed axes (reviewer requirement)
    # ---------------------------------------------------------
    ax.set_title(title)
    ax.set_xlabel("RAM Consumption (KB)")
    ax.set_ylabel("Val. Accuracy")

    ax.set_ylim(0.3, 0.75)
    ax.set_xlim(1200, 0)
    ax.grid(True, alpha=0.35, linewidth=1.0)

    # ---------------------------------------------------------
    # Legend box 1 (top): Flash Memory Range (fixed 0..1100 KB)
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
        ax_leg.scatter([], [], s=s, color="tab:blue", alpha=0.85,
                       edgecolors="black", linewidths=1.2, label=lab)
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
    # Legend box 2 (middle): Runs
    # ---------------------------------------------------------
    run_handles = [
        Line2D([0], [0],
               marker=mk, color="w",
               label=txt,
               markerfacecolor=col,
               markeredgecolor="black",
               markersize=8)
        for (txt, col, mk) in legend_entries
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
    # Legend box 3 (bottom): MCU budgets (only constrained)
    # ---------------------------------------------------------
    if is_constrained:
        mcu_devices = [
            ("Arduino Nano 33 BLE",     256, 1024),
            ("STM32F411 (Nucleo)",      128, 512),
            ("Raspberry Pi Pico",       264, 2048),
            ("ESP32-C3 DevKit",         400, 4096),
        ]

        mcu_handles = []
        for name, ram_kb, flash_kb in mcu_devices:
            label = f"{name}: {ram_kb} KB RAM, {format_flash_size(flash_kb)} Flash"
            mcu_handles.append(
                Line2D([0], [0],
                       marker="o", linestyle="",
                       color="gray",
                       markerfacecolor="none",
                       markeredgecolor="gray",
                       label=label)
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
    # Hypervolume annotation (bottom-left inside plot)
    # ---------------------------------------------------------
    if hv_with is not None:
        text = (
            f"Hv No RankNet: {hv_without:.3f}\n"
            f"Hv RankNet : {hv_with:.3f}\n"
            f"ΔHv Improvement: {improvement:.1f}%"
        )
        ax.text(
            0.02, 0.02, text,
            transform=ax.transAxes,
            fontsize=11,
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
    args = parser.parse_args()

    plot_hour_run_new(
        root_dir=args.hour_run_dir,
        title=args.title,
        marker_scale=args.marker_scale,
        out_path=args.out,
    )
