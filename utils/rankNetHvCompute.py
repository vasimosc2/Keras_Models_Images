import os
import argparse
import pandas as pd
import numpy as np


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

        ram = pd.to_numeric(df[ram_col], errors="coerce").to_numpy(dtype=float)
        acc = pd.to_numeric(df[acc_col], errors="coerce").to_numpy(dtype=float)
        flash = pd.to_numeric(df[flash_col], errors="coerce").to_numpy(dtype=float)

        valid = ~np.isnan(ram) & ~np.isnan(acc) & ~np.isnan(flash)
        if np.any(valid):
            pts.append(np.column_stack([ram[valid], acc[valid], flash[valid]]))

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


def accuracy_scale(points: np.ndarray):
    """
    Detect whether accuracy is represented as:
      - fraction in [0,1]
      - percentage in [0,100]

    Returns:
      1.0 or 100.0
    """
    if points is None or points.size == 0:
        return 1.0

    acc_max = np.nanmax(points[:, 1])
    if acc_max <= 1.5:
        return 1.0
    return 100.0


def normalize_two_sets(points_a: np.ndarray, points_b: np.ndarray, margin=0.05):
    """
    Jointly normalize RAM and FLASH across the two compared sets.

    The normalization upper bound is expanded by `margin`, so the worst observed
    RAM/FLASH does not map exactly to 1.0.

    Accuracy is kept in its original representation and later converted to MISS.

    Returns:
      norm_a, norm_b
    """
    if points_a is None or len(points_a) == 0:
        raise ValueError("points_a is empty")
    if points_b is None or len(points_b) == 0:
        raise ValueError("points_b is empty")

    all_pts = np.vstack([points_a, points_b])

    ram_min = np.nanmin(all_pts[:, 0])
    ram_max_obs = np.nanmax(all_pts[:, 0])

    flash_min = np.nanmin(all_pts[:, 2])
    flash_max_obs = np.nanmax(all_pts[:, 2])

    # Expanded worst point
    ram_max = ram_max_obs * (1.0 + margin)
    flash_max = flash_max_obs * (1.0 + margin)

    ram_range = ram_max - ram_min
    flash_range = flash_max - flash_min

    def norm(pts):
        ram = pts[:, 0]
        acc = pts[:, 1]
        flash = pts[:, 2]

        ram_norm = (ram - ram_min) / (ram_range + 1e-12)
        flash_norm = (flash - flash_min) / (flash_range + 1e-12)

        return np.column_stack([ram_norm, acc, flash_norm])

    return norm(points_a), norm(points_b)


def to_normalized_minimization(points_norm: np.ndarray) -> np.ndarray:
    """
    Convert normalized points into minimization costs:

      - RAM cost   = normalized RAM in [0,1]
      - MISS cost  = 1 - ACC   if ACC in [0,1]
                     (100 - ACC)/100 if ACC in [0,100]
      - FLASH cost = normalized Flash in [0,1]

    Returns:
      costs: shape (N,3)
    """
    if points_norm is None or len(points_norm) == 0:
        return np.empty((0, 3))

    ram_cost = points_norm[:, 0]
    acc = points_norm[:, 1]
    flash_cost = points_norm[:, 2]

    acc_scale = accuracy_scale(points_norm)

    if acc_scale == 1.0:
        miss_cost = 1.0 - acc
    else:
        miss_cost = (100.0 - acc) / 100.0

    return np.column_stack([ram_cost, miss_cost, flash_cost])


def hypervolume_2d_min(cost_pts: np.ndarray, ref=(1.0, 1.0)) -> float:
    """
    Hypervolume for 2D minimization w.r.t. reference point.
    """
    if cost_pts.size == 0:
        return 0.0

    r1, r2 = ref
    pts = np.asarray(cost_pts, dtype=float)

    mask = (pts[:, 0] <= r1) & (pts[:, 1] <= r2)
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

        intervals = np.column_stack([
            pts[slab_mask, 1],
            np.full(np.sum(slab_mask), r2)
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


def theoretical_hypervolume():
    """
    Theoretical HV of the normalized unit cube.
    """
    return 1.0


def compute_hv_for_normalized_points(points_norm: np.ndarray):
    """
    Compute HV results for one normalized set using ref=(1,1,1).
    """
    if points_norm is None or len(points_norm) == 0:
        return None

    pts_pf = pareto_front_3obj(points_norm)
    cost_pf = to_normalized_minimization(pts_pf)

    ref = (1.0, 1.0, 1.0)
    total_theoretical_hv = theoretical_hypervolume()

    hv_pf = hypervolume_3d_min(cost_pf, ref=ref)
    coverage_pf = hv_pf * 100.0

    # Per-model single HVs (for all models, not only Pareto points)
    all_costs = to_normalized_minimization(points_norm)
    model_coverages = []
    for idx, c in enumerate(all_costs):
        hv_single = hypervolume_3d_min(np.array([c]), ref=ref)
        cov_single = hv_single * 100.0

        model_coverages.append({
            "model_index": idx + 1,
            "ram_norm": points_norm[idx, 0],
            "acc": points_norm[idx, 1],
            "flash_norm": points_norm[idx, 2],
            "hv_single": hv_single,
            "coverage_percent": cov_single,
        })

    return {
        "ref": ref,
        "theoretical_hv": total_theoretical_hv,
        "hv_pf": hv_pf,
        "coverage_pf_percent": coverage_pf,
        "pareto_points": pts_pf,
        "model_coverages": model_coverages,
        "num_models": len(points_norm),
        "num_pareto_models": len(pts_pf),
    }


def compute_combined_hypervolume(with_data, without_data):
    """
    Combine all WithoutRankNet CSVs into one group and all WithRankNet CSVs into one group,
    then compare their hypervolumes using JOINT normalization.

    The theoretical HV is always 1.
    """
    pts_with = collect_ram_acc_flash(with_data)
    pts_without = collect_ram_acc_flash(without_data)

    if pts_with is None or pts_without is None:
        return None

    norm_without, norm_with = normalize_two_sets(pts_without, pts_with)

    without_res = compute_hv_for_normalized_points(norm_without)
    with_res = compute_hv_for_normalized_points(norm_with)

    hv_with = with_res["hv_pf"]
    hv_without = without_res["hv_pf"]
    improvement = format_improvement(hv_without, hv_with)

    return {
        "hv_without": hv_without,
        "hv_with": hv_with,
        "improvement": improvement,
        "coverage_without_percent": without_res["coverage_pf_percent"],
        "coverage_with_percent": with_res["coverage_pf_percent"],
        "theoretical_hv": 1.0,
        "ref": (1.0, 1.0, 1.0),
        "without_details": without_res,
        "with_details": with_res,
    }


def compute_all_vs_all_hypervolume(with_data, without_data):
    """
    Compare every WithoutRankNet file against every WithRankNet file
    using JOINT normalization for each pair.

    The theoretical HV is always 1.
    """
    results = []

    for i, (without_fname, without_df) in enumerate(without_data, start=1):
        pts_without = collect_ram_acc_flash([(without_fname, without_df)])
        if pts_without is None:
            continue

        for j, (with_fname, with_df) in enumerate(with_data, start=1):
            pts_with = collect_ram_acc_flash([(with_fname, with_df)])
            if pts_with is None:
                continue

            norm_without, norm_with = normalize_two_sets(pts_without, pts_with)

            without_res = compute_hv_for_normalized_points(norm_without)
            with_res = compute_hv_for_normalized_points(norm_with)

            hv_without = without_res["hv_pf"]
            hv_with = with_res["hv_pf"]
            improvement = format_improvement(hv_without, hv_with)

            results.append({
                "without_index": i,
                "with_index": j,
                "without_file": without_fname,
                "with_file": with_fname,
                "hv_without": hv_without,
                "hv_with": hv_with,
                "improvement": improvement,
                "coverage_without_percent": without_res["coverage_pf_percent"],
                "coverage_with_percent": with_res["coverage_pf_percent"],
                "theoretical_hv": 1.0,
                "ref": (1.0, 1.0, 1.0),
                "without_details": without_res,
                "with_details": with_res,
            })

    return results


def print_top_model_coverages(label, details, top_k=5):
    """
    Print the top-k single-model theoretical HV coverages for a run/group.
    Uses normalized RAM and Flash.
    """
    if details is None or not details["model_coverages"]:
        print(f"      No model coverages available for {label}.")
        return

    ranked = sorted(
        details["model_coverages"],
        key=lambda x: x["coverage_percent"],
        reverse=True
    )[:top_k]

    print(f"      Top {len(ranked)} single-model coverages {label}:")
    for m in ranked:
        print(
            f"         Model {m['model_index']}: "
            f"RAM_norm={m['ram_norm']:.6f}, "
            f"ACC={m['acc']:.6f}, "
            f"FLASH_norm={m['flash_norm']:.6f}, "
            f"HV={m['hv_single']:.6f}, "
            f"Coverage={m['coverage_percent']:.4f}%"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute normalized hypervolume for RankNet vs non-RankNet runs."
    )
    parser.add_argument(
        "--hour_run_dir",
        type=str,
        required=True,
        help='Path to the hour-run folder (must contain "WithRankNet" and/or "WithoutRankNet").',
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
    parser.add_argument(
        "--top_k_models",
        type=int,
        default=5,
        help="How many top single-model coverages to print.",
    )
    args = parser.parse_args()

    combine_runs = str_to_bool(
        args.combine_runs if args.combine_runs is not None else os.getenv("COMBINE_RUNS"),
        default=False
    )

    with_dir = os.path.join(args.hour_run_dir, "WithRankNet")
    without_dir = os.path.join(args.hour_run_dir, "WithoutRankNet")

    with_ranknet_data = read_csvs_from_folder(with_dir)
    without_ranknet_data = read_csvs_from_folder(without_dir)

    if not with_ranknet_data and not without_ranknet_data:
        raise SystemExit(f"No CSVs found in {with_dir} or {without_dir}")

    print(f"\n📁 Evaluating folder: {args.hour_run_dir}")
    print(f"⚙️ combine_runs = {combine_runs}")
    print("📦 Reference box mode: normalized per comparison")
    print("   - RAM normalized to [0,1] jointly across the 2 compared runs")
    print("   - FLASH normalized to [0,1] jointly across the 2 compared runs")
    print("   - MISS = 1 - ACC")
    print("   - Theoretical HV = 1.0\n")

    if with_ranknet_data and without_ranknet_data:
        if combine_runs:
            combined_result = compute_combined_hypervolume(
                with_ranknet_data,
                without_ranknet_data,
            )

            if combined_result is None:
                print("⚠️ No valid data found for combined comparison.\n")
            else:
                ref = combined_result["ref"]
                print("📊 Combined Hypervolume Comparison (normalized unit cube):")
                print(f"   Reference box      : RAM={ref[0]:.4f}, MISS={ref[1]:.4f}, FLASH={ref[2]:.4f}")
                print(f"   Theoretical HV     : {combined_result['theoretical_hv']:.10f}\n")

                print(f"   HV WITHOUT RankNet : {combined_result['hv_without']:.10f}")
                print(f"   Coverage WITHOUT   : {combined_result['coverage_without_percent']:.6f}%\n")

                print(f"   HV WITH RankNet    : {combined_result['hv_with']:.10f}")
                print(f"   Coverage WITH      : {combined_result['coverage_with_percent']:.6f}%\n")

                print(f"   Improvement        : {combined_result['improvement']}\n")

                print_top_model_coverages("WITHOUT", combined_result["without_details"], top_k=args.top_k_models)
                print()
                print_top_model_coverages("WITH", combined_result["with_details"], top_k=args.top_k_models)
                print()

        else:
            all_vs_all_results = compute_all_vs_all_hypervolume(
                with_ranknet_data,
                without_ranknet_data,
            )

            print("📊 All-vs-All Hypervolume Comparison (normalized unit cube):")
            for res in all_vs_all_results:
                ref = res["ref"]

                print(
                    f"   Without Run {res['without_index']} ({res['without_file']}) "
                    f"vs With Run {res['with_index']} ({res['with_file']})"
                )
                print(f"      Reference box      : RAM={ref[0]:.4f}, MISS={ref[1]:.4f}, FLASH={ref[2]:.4f}")
                print(f"      Theoretical HV     : {res['theoretical_hv']:.10f}")

                print(f"      HV WITHOUT RankNet : {res['hv_without']:.10f}")
                print(f"      Coverage WITHOUT   : {res['coverage_without_percent']:.6f}%")

                print(f"      HV WITH RankNet    : {res['hv_with']:.10f}")
                print(f"      Coverage WITH      : {res['coverage_with_percent']:.6f}%")

                print(f"      Improvement        : {res['improvement']}")

                print_top_model_coverages("WITHOUT", res["without_details"], top_k=args.top_k_models)
                print_top_model_coverages("WITH", res["with_details"], top_k=args.top_k_models)
                print()
    else:
        print("⚠️ Hypervolume comparison skipped because one of the folders is empty.\n")