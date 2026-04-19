import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path

# Markers to distinguish different runs inside the same group
MARKERS = ["o", "s", "D", "^", "v", "P", "X"]


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
    return f"{mb:.1f} MB"


def compute_auto_ram_limit(size_info, default_limit=1200.0, margin_ratio=0.10, min_limit=10.0):
    """
    Automatically compute a reasonable RAM x-axis maximum from the data.

    It finds the largest RAM value across all runs and adds a margin.
    """
    all_ram_values = []

    for _, fname, df, _ in size_info:
        if "Model RAM (KB)" not in df.columns:
            raise ValueError(f"{fname} is missing column 'Model RAM (KB)'")

        ram_vals = pd.to_numeric(df["Model RAM (KB)"], errors="coerce").dropna().to_numpy()
        if ram_vals.size > 0:
            all_ram_values.extend(ram_vals.tolist())

    if not all_ram_values:
        return default_limit

    data_ram_max = max(all_ram_values)
    ram_limit = data_ram_max * (1.0 + margin_ratio)
    ram_limit = max(min_limit, ram_limit)

    return ram_limit


def plot_hour_run(
    root_dir,
    title=None,
    marker_scale=1.0,
    out_path=None,
    ram_max=None,
    auto_ram_margin=0.10,
):
    """
    Plot RankNet vs non-RankNet runs from one hour-run folder.

    This script:
      - does NOT compute hypervolume
      - does NOT annotate hypervolume
      - supports either:
          * manual RAM axis limit via ram_max
          * automatic RAM axis limit from the data
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

    with_dir = os.path.join(root_dir, "WithRankNet")
    without_dir = os.path.join(root_dir, "WithoutRankNet")

    color_with = "tab:blue"
    color_without = "tab:orange"

    with_ranknet_data = read_csvs_from_folder(with_dir)
    without_ranknet_data = read_csvs_from_folder(without_dir)

    if not with_ranknet_data and not without_ranknet_data:
        raise SystemExit(f"No CSVs found in {with_dir} or {without_dir}")

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

    # Gather inputs
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

    # Automatic RAM scaling if not provided
    if ram_max is None:
        ram_max = compute_auto_ram_limit(size_info, margin_ratio=auto_ram_margin)

    print(f"Using RAM x-axis max: {ram_max:.2f} KB")

    # Figure layout
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
            alpha=0.90,
            color=color_with,
            marker=marker,
            edgecolors="black",
            linewidths=0.7,
            zorder=3,
        )

        run_id_with += 1
        legend_entries.append((f"With RankNet - Run {run_id_with}", color_with, marker))

    # Axes
    ax.set_title(title)
    ax.set_xlabel("RAM Consumption (KB)")
    ax.set_ylabel("Val. Accuracy")

    ax.set_ylim(0.3, 0.75)
    ax.set_xlim(ram_max, 0)
    ax.grid(True, alpha=0.35, linewidth=1.0)

    # Legend box 1: Flash Memory Range
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

    # Legend box 2: Runs
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

    # Legend box 3: MCU budgets (only constrained)
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

    fig.savefig(out_path, bbox_inches="tight")
    print(f"✅ saved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot RankNet vs non-RankNet runs from a single hour-run folder."
    )
    parser.add_argument(
        "--hour_run_dir",
        type=str,
        required=True,
        help='Path to the hour-run folder (e.g. "./ThesisResults/Constrained/3Hours"). This folder must contain "WithRankNet" and/or "WithoutRankNet".',
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
        "--ram_max",
        type=float,
        default=None,
        help="Manual maximum RAM value to show on the x-axis. If omitted, automatic scaling is used.",
    )
    parser.add_argument(
        "--auto_ram_margin",
        type=float,
        default=0.10,
        help="Extra margin added to the automatic RAM max. Example: 0.10 means 10%%.",
    )
    args = parser.parse_args()

    plot_hour_run(
        root_dir=args.hour_run_dir,
        title=args.title,
        marker_scale=args.marker_scale,
        out_path=args.out,
        ram_max=args.ram_max,
        auto_ram_margin=args.auto_ram_margin,
    )