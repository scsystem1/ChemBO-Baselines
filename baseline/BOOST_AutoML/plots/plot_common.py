import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogFormatterSciNotation

N_REPS = 30
DEFAULT_LOG_FLOOR = np.finfo(float).tiny
LOWER_QUANTILE = 0.1
MAX_SAMPLES = 100
INITIAL_POINTS = 10
PLOT_START_INDEX = INITIAL_POINTS - 1
PLOTTED_SAMPLES = MAX_SAMPLES - PLOT_START_INDEX
OPTIMUM_THRESHOLD = 1e-10
SYNTHETIC_4D = {"Ackley", "Levy", "Rosenbrock", "SumSquares"}
TARGET_SELECTIONS = {
    "Ackley",
    "Levy",
    "Rosenbrock",
    "AgNP",
    "P3HT",
    "2277",
    "5527",
    "5636",
    "5891",
    "5906",
    "5964",
    "6322",
    "6762",
    "6794",
    "7200",
}
SPECIAL_STYLES = {
    "BOOST": {"color": "#d62728", "linewidth": 4.5, "linestyle": "-", "zorder": 5},
    "Random": {"color": "#7f7f7f", "linewidth": 3.2, "linestyle": ":", "zorder": 3},
    "Static": {"color": "#9467bd", "linewidth": 3.2, "linestyle": "-", "zorder": 4},
    "Adaptive Kernel": {"color": "#2ca02c", "linewidth": 3.2, "linestyle": "-", "zorder": 4},
    "Adaptive Acquisition": {"color": "#ff7f0e", "linewidth": 3.2, "linestyle": "-", "zorder": 4},
    "Matérn 3/2 + EI": {"color": "#8B4513", "linewidth": 3.2, "linestyle": "--", "zorder": 4},
    # External and diagnostic baselines kept distinct from the existing BOOST/Random/Static palette.
    "qLogNEI (q=1)": {"color": "#1f77b4", "linewidth": 3.2, "linestyle": "-", "zorder": 4},
    "BOOST (fixed HP)": {"color": "#d62728", "linewidth": 3.2, "linestyle": "--", "zorder": 4},
    "HEBO": {"color": "#17becf", "linewidth": 3.2, "linestyle": "-", "zorder": 4},
    # Random-KA: magenta dash-dot, kept distinct from grey-dotted Random Search.
    "Random-KA": {"color": "#e377c2", "linewidth": 3.2, "linestyle": "-.", "zorder": 3},
}

FIXED_ORDER = [
    ("Matern32", "EI"),
    ("Matern52", "EI"),
    ("RBF", "EI"),
    ("RQ", "EI"),
    ("Matern32", "PI"),
    ("Matern52", "PI"),
    ("RBF", "PI"),
    ("RQ", "PI"),
    ("Matern32", "UCB"),
    ("Matern52", "UCB"),
    ("RBF", "UCB"),
    ("RQ", "UCB"),
    ("Matern32", "PM"),
    ("Matern52", "PM"),
    ("RBF", "PM"),
    ("RQ", "PM"),
]


def parse_objective_name(folder_name: str) -> str:
    if folder_name in SYNTHETIC_4D:
        return folder_name
    if re.fullmatch(r"\d{2}_.+", folder_name):
        return folder_name.split("_", 1)[1]
    return folder_name


def parse_title(folder_name: str, objective_name: str, objective_stem: str | None = None) -> str:
    match = re.fullmatch(r"(\d+)_(\d+D)", folder_name)
    if match:
        search_id, dim = match.groups()
        return f"Search Space ID: {search_id} ({dim})"
    if objective_stem is not None:
        stem_match = re.fullmatch(r"(.+)_([0-9]+D)", objective_stem)
        if stem_match:
            _, dim = stem_match.groups()
            return f"Search Space ID: {objective_name} ({dim})"
    if objective_name in SYNTHETIC_4D:
        return f"Search Space ID: {objective_name} (4D)"
    return f"Search Space ID: {objective_name}"


def matches_target_selection(folder_name: str) -> bool:
    if folder_name in TARGET_SELECTIONS:
        return True
    objective_name = parse_objective_name(folder_name)
    if objective_name in TARGET_SELECTIONS:
        return True
    match = re.fullmatch(r"(\d+)_.+", folder_name)
    return bool(match and match.group(1) in TARGET_SELECTIONS)


def infer_objective_stem(objective_dir: Path, objective_name: str) -> str:
    fixed_dir = objective_dir / "Fixed_ker_acq"
    for path in sorted(fixed_dir.glob("*_results.xlsx")):
        stem = path.stem
        for kernel, acquisition in FIXED_ORDER:
            suffix = f"_{kernel}_{acquisition}_results"
            if stem.endswith(suffix):
                return stem[: -len(suffix)]

    boost_path = find_results_file(objective_dir / "BOOST", "recommended_results")
    if boost_path is not None:
        stem = boost_path.stem
        suffix = "_recommended_results"
        if stem.endswith(suffix):
            return stem[: -len(suffix)]

    return objective_name


def find_results_file(directory: Path, pattern: str | None = None) -> Path | None:
    if not directory.exists():
        return None
    files = sorted(directory.glob("*_results.xlsx"))
    if pattern is not None:
        files = [path for path in files if pattern in path.name]
    return files[0] if files else None


def load_statistics(path: Path) -> dict:
    df = pd.read_excel(path, sheet_name="statistics", header=None)
    iterations = df.iloc[0, 1 + PLOT_START_INDEX : 1 + MAX_SAMPLES].to_numpy(dtype=float) + 1
    means = df.iloc[1, 1 + PLOT_START_INDEX : 1 + MAX_SAMPLES].to_numpy(dtype=float)
    stds = df.iloc[2, 1 + PLOT_START_INDEX : 1 + MAX_SAMPLES].to_numpy(dtype=float)
    return {"iterations": iterations, "means": means, "stds": stds}


def load_lower_quantile(path: Path) -> dict:
    df = pd.read_excel(path, sheet_name="combined_seeds", header=None)
    iterations = df.iloc[0, 1 + PLOT_START_INDEX : 1 + MAX_SAMPLES].to_numpy(dtype=float) + 1
    seed_values = df.iloc[1:, 1:1 + MAX_SAMPLES].to_numpy(dtype=float)
    quantile = np.quantile(seed_values, 1 - LOWER_QUANTILE, axis=0)[PLOT_START_INDEX:MAX_SAMPLES]
    return {"iterations": iterations, "quantile": quantile}


def load_samples_to_optimum(path: Path) -> np.ndarray:
    df = pd.read_excel(path, sheet_name="combined_seeds", header=None)
    seed_values = df.iloc[1:, 1:1 + MAX_SAMPLES].to_numpy(dtype=float)
    hit_samples = []
    for row in seed_values:
        hit_indices = np.where(row <= OPTIMUM_THRESHOLD)[0]
        if len(hit_indices):
            hit_iter = max(int(hit_indices[0]) + 1, INITIAL_POINTS)
        else:
            hit_iter = MAX_SAMPLES
        hit_samples.append(hit_iter)
    return np.asarray(hit_samples, dtype=float)


def load_final_regrets(path: Path) -> np.ndarray:
    df = pd.read_excel(path, sheet_name="combined_seeds", header=None)
    seed_values = df.iloc[1:, 1:1 + MAX_SAMPLES].to_numpy(dtype=float)
    return seed_values[:, MAX_SAMPLES - 1]


def load_gap_closed(path: Path) -> dict:
    df = pd.read_excel(path, sheet_name="combined_seeds", header=None)
    iterations = df.iloc[0, 1 + PLOT_START_INDEX : 1 + MAX_SAMPLES].to_numpy(dtype=float) + 1
    seed_values = df.iloc[1:, 1:1 + MAX_SAMPLES].to_numpy(dtype=float)
    initial = seed_values[:, [0]]
    gap_closed = np.divide(
        initial - seed_values,
        initial,
        out=np.ones_like(seed_values, dtype=float),
        where=np.abs(initial) > OPTIMUM_THRESHOLD,
    )
    gap_closed = np.clip(gap_closed, 0.0, 1.0)
    return {
        "iterations": iterations,
        "means": np.mean(gap_closed, axis=0)[PLOT_START_INDEX:MAX_SAMPLES],
        "stds": np.std(gap_closed, axis=0)[PLOT_START_INDEX:MAX_SAMPLES],
    }


def compute_regret_floor(value_arrays: list[np.ndarray]) -> float:
    all_values = []
    for values in value_arrays:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        arr[np.abs(arr) <= OPTIMUM_THRESHOLD] = 0.0
        if arr.size:
            all_values.append(arr)

    if not all_values:
        return DEFAULT_LOG_FLOOR

    merged = np.concatenate(all_values)
    positive = merged[merged > 0]
    if positive.size == 0:
        return DEFAULT_LOG_FLOOR

    reference = float(np.min(positive))
    zero_present = np.any(merged == 0)
    multiplier = 0.1 if zero_present else 0.8
    return float(max(DEFAULT_LOG_FLOOR, multiplier * reference))


def render_log_curve(raw_values: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw_values, dtype=float)
    raw[np.abs(raw) <= OPTIMUM_THRESHOLD] = 0.0
    rendered = raw.copy()
    rendered[~np.isfinite(rendered)] = np.nan
    zero_indices = np.where(np.isfinite(raw) & (raw <= 0))[0]
    if len(zero_indices):
        first_zero = int(zero_indices[0])
        rendered[first_zero] = floor
        if first_zero + 1 < len(rendered):
            rendered[first_zero + 1:] = np.nan
        zero_mask = np.zeros_like(raw, dtype=bool)
        zero_mask[first_zero] = True
    else:
        zero_mask = np.zeros_like(raw, dtype=bool)
    return rendered, zero_mask


def render_log_band(raw_values: np.ndarray, floor: float, stop_after: np.ndarray | None = None) -> np.ndarray:
    rendered = np.asarray(raw_values, dtype=float).copy()
    rendered[np.abs(rendered) <= OPTIMUM_THRESHOLD] = 0.0
    rendered[~np.isfinite(rendered)] = np.nan
    finite_negative = np.isfinite(rendered) & (rendered <= 0)
    rendered[finite_negative] = floor
    if stop_after is not None:
        rendered[stop_after] = np.nan
    return rendered


def render_log_box_values(raw_values: np.ndarray, floor: float) -> np.ndarray:
    rendered = np.asarray(raw_values, dtype=float).copy()
    rendered[np.abs(rendered) <= OPTIMUM_THRESHOLD] = 0.0
    rendered[~np.isfinite(rendered)] = np.nan
    rendered[np.isfinite(rendered) & (rendered <= 0)] = floor
    return rendered


def apply_bottom_tick(ax, floor: float) -> None:
    formatter = LogFormatterSciNotation()
    _, top = ax.get_ylim()
    ticks = [tick for tick in ax.get_yticks() if floor < tick <= top]
    ax.set_yticks([floor] + ticks)
    ax.yaxis.set_major_formatter(formatter)


def format_combo_label(kernel: str, acquisition: str) -> str:
    kernel_label = {
        "Matern32": "Matérn 3/2",
        "Matern52": "Matérn 5/2",
        "RBF": "RBF",
        "RQ": "RQ",
    }[kernel]
    acq_label = "LCB" if acquisition == "UCB" else acquisition
    return f"{kernel_label} + {acq_label}"


def iter_fixed_results(objective_dir: Path, objective_stem: str):
    fixed_dir = objective_dir / "Fixed_ker_acq"
    for kernel, acquisition in FIXED_ORDER:
        path = fixed_dir / f"{objective_stem}_{kernel}_{acquisition}_results.xlsx"
        if path.exists():
            yield kernel, acquisition, path


def plot_with_ci(ax, stats: dict, label: str, color: str, linewidth: float, linestyle: str, zorder: int, floor: float) -> None:
    x = stats["iterations"]
    raw_y = np.asarray(stats["means"], dtype=float)
    ci95 = 1.96 * stats["stds"] / math.sqrt(N_REPS)
    y, zero_mask = render_log_curve(raw_y, floor)
    stop_after = np.zeros_like(raw_y, dtype=bool)
    if np.any(zero_mask):
        first_zero = int(np.where(zero_mask)[0][0])
        if first_zero + 1 < len(stop_after):
            stop_after[first_zero + 1:] = True
    lower = render_log_band(raw_y - ci95, floor, stop_after=stop_after)
    upper = render_log_band(raw_y + ci95, floor, stop_after=stop_after)
    ax.plot(x, y, label=label, color=color, linewidth=linewidth, linestyle=linestyle, zorder=zorder)
    ax.fill_between(
        x,
        lower,
        upper,
        where=np.isfinite(lower) & np.isfinite(upper),
        color=color,
        alpha=0.18 if label == "BOOST" else (0.12 if linewidth >= 4 else 0.07),
        zorder=zorder - 1,
    )
    if np.any(zero_mask):
        ax.scatter(x[zero_mask], np.full(np.sum(zero_mask), floor), marker="v", s=18, color=color, zorder=zorder + 1, clip_on=False)


def plot_quantile(ax, quantile_stats: dict, label: str, color: str, linewidth: float, linestyle: str, zorder: int, floor: float) -> None:
    x = quantile_stats["iterations"]
    raw_y = np.asarray(quantile_stats["quantile"], dtype=float)
    y, zero_mask = render_log_curve(raw_y, floor)
    ax.plot(x, y, label=label, color=color, linewidth=linewidth, linestyle=linestyle, zorder=zorder)
    if np.any(zero_mask):
        ax.scatter(x[zero_mask], np.full(np.sum(zero_mask), floor), marker="v", s=18, color=color, zorder=zorder + 1, clip_on=False)


def plot_remaining_gap(ax, gap_stats: dict, label: str, color: str, linewidth: float, linestyle: str, zorder: int, floor: float) -> None:
    x = gap_stats["iterations"]
    raw_y = 1.0 - np.asarray(gap_stats["means"], dtype=float)
    ci95 = 1.96 * gap_stats["stds"] / math.sqrt(N_REPS)
    y, zero_mask = render_log_curve(raw_y, floor)
    stop_after = np.zeros_like(raw_y, dtype=bool)
    if np.any(zero_mask):
        first_zero = int(np.where(zero_mask)[0][0])
        if first_zero + 1 < len(stop_after):
            stop_after[first_zero + 1:] = True
    lower = render_log_band(raw_y - ci95, floor, stop_after=stop_after)
    upper = render_log_band(raw_y + ci95, floor, stop_after=stop_after)
    ax.plot(x, y, label=label, color=color, linewidth=linewidth, linestyle=linestyle, zorder=zorder)
    ax.fill_between(
        x,
        lower,
        upper,
        where=np.isfinite(lower) & np.isfinite(upper),
        color=color,
        alpha=0.18 if label == "BOOST" else (0.12 if linewidth >= 4 else 0.07),
        zorder=zorder - 1,
    )
    if np.any(zero_mask):
        ax.scatter(x[zero_mask], np.full(np.sum(zero_mask), floor), marker="v", s=18, color=color, zorder=zorder + 1, clip_on=False)


def plot_boxplot(ax, entries: list[dict], title: str, ylabel: str, log_scale: bool, floor: float | None = None) -> None:
    values = []
    for entry in entries:
        arr = np.asarray(entry["values"], dtype=float)
        if log_scale and floor is not None:
            arr = render_log_box_values(arr, floor)
        values.append(arr)
    labels = [entry["label"] for entry in entries]
    colors = [entry["color"] for entry in entries]
    bp = ax.boxplot(
        values,
        patch_artist=True,
        vert=True,
        widths=0.62,
        whis=(0, 100),
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.8},
        whiskerprops={"linewidth": 1.4},
        capprops={"linewidth": 1.4},
        boxprops={"linewidth": 1.2},
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
        patch.set_edgecolor(color)

    repeated_colors = [color for color in colors for _ in range(2)]
    for whisker, color in zip(bp["whiskers"], repeated_colors):
        whisker.set_color(color)
    for cap, color in zip(bp["caps"], repeated_colors):
        cap.set_color(color)

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12)
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(bottom=floor if floor is not None else DEFAULT_LOG_FLOOR)
        if floor is not None:
            apply_bottom_tick(ax, floor)
    else:
        finite_values = [np.asarray(v, dtype=float) for v in values]
        finite_values = [v[np.isfinite(v)] for v in finite_values if np.any(np.isfinite(v))]
        top = None
        if finite_values:
            top = float(max(np.max(v) for v in finite_values))
        if "Evaluations to Optimum" in ylabel:
            ax.set_ylim(INITIAL_POINTS, MAX_SAMPLES)
        else:
            ax.set_ylim(0, 1.05 * top if top is not None and top > 0 else 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="y", labelsize=10)


def sort_entries(entries: list[dict]) -> list[dict]:
    boost_entries = [entry for entry in entries if entry["label"] == "BOOST"]
    others = [entry for entry in entries if entry["label"] != "BOOST"]

    def entry_key(entry: dict):
        values = np.asarray(entry["values"], dtype=float)
        median = float(np.median(values))
        q1, q3 = np.percentile(values, [25, 75])
        return (median, float(q3 - q1), entry["label"])

    others.sort(key=entry_key)
    return boost_entries + others


def style_regret_axis(ax, title: str, floor: float, top: float | None = None) -> None:
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Evaluation", fontsize=12)
    ax.set_ylabel("Regret", fontsize=12)
    ax.set_yscale("log")
    if top is not None:
        ax.set_ylim(bottom=floor, top=top)
    else:
        ax.set_ylim(bottom=floor)
    ax.set_xlim(INITIAL_POINTS, MAX_SAMPLES)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    apply_bottom_tick(ax, floor)


def style_remaining_gap_axis(ax, title: str, floor: float, top: float | None = None) -> None:
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Evaluation", fontsize=12)
    ax.set_ylabel("Remaining Gap", fontsize=12)
    ax.set_yscale("log")
    ax.set_xlim(INITIAL_POINTS, MAX_SAMPLES)
    if top is not None:
        ax.set_ylim(bottom=floor, top=top)
    else:
        ax.set_ylim(bottom=floor)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    apply_bottom_tick(ax, floor)


def build_special_sources(objective_dir: Path, objective_stem: str) -> dict[str, Path]:
    fixed_dir = objective_dir / "Fixed_ker_acq"
    sources = {}
    boost_path = find_results_file(objective_dir / "BOOST", "recommended_results")
    random_path = find_results_file(objective_dir / "Random")
    static_path = find_results_file(objective_dir / "Static")
    adaptive_kernel_path = find_results_file(objective_dir / "Adaptive_ker")
    adaptive_acq_path = find_results_file(objective_dir / "Adaptive_acq")
    baseline_path = fixed_dir / f"{objective_stem}_Matern32_EI_results.xlsx"

    # External and diagnostic baselines live in sibling subdirectories under the task
    # folder, mirroring the existing BOOST / Random / Static layout. They are
    # optional: if the directory is missing the baseline is silently skipped, so
    # existing figures regenerate unchanged when the new results are not yet present.
    botorch_default_path = find_results_file(objective_dir / "BoTorch_Default")
    boost_fixedhp_path = find_results_file(objective_dir / "BOOST_FixedHP", "recommended_results")
    hebo_path = find_results_file(objective_dir / "HEBO", "HEBO_MACE_results")
    random_kernel_af_path = find_results_file(objective_dir / "Random_KernelAF", "recommended_results")

    if boost_path is not None:
        sources["BOOST"] = boost_path
    if random_path is not None:
        sources["Random"] = random_path
    if static_path is not None:
        sources["Static"] = static_path
    if adaptive_kernel_path is not None:
        sources["Adaptive Kernel"] = adaptive_kernel_path
    if adaptive_acq_path is not None:
        sources["Adaptive Acquisition"] = adaptive_acq_path
    if baseline_path.exists():
        sources[format_combo_label("Matern32", "EI")] = baseline_path
    if botorch_default_path is not None:
        sources["qLogNEI (q=1)"] = botorch_default_path
    if boost_fixedhp_path is not None:
        sources["BOOST (fixed HP)"] = boost_fixedhp_path
    if hebo_path is not None:
        sources["HEBO"] = hebo_path
    if random_kernel_af_path is not None:
        sources["Random-KA"] = random_kernel_af_path
    return sources


def build_result_bundle(path: Path) -> dict:
    return {
        "stats": load_statistics(path),
        "quantile": load_lower_quantile(path),
        "samples_to_optimum": load_samples_to_optimum(path),
        "final_regrets": load_final_regrets(path),
        "gap_closed": load_gap_closed(path),
    }
