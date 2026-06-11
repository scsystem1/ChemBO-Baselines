from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.ticker import LogFormatterSciNotation
from plot_config import ABLATION_FIGURES_DIR, ABLATION_RESULTS_DIR, RESULTS_DIR, save_figure

OUTPUT_DIR = ABLATION_FIGURES_DIR
OBJECTIVES = ["Ackley", "Levy", "Rosenbrock"]
DEFAULT_OBJECTIVE_DIRS = {
    "Ackley": RESULTS_DIR / "01_Ackley" / "BOOST",
    "Levy": RESULTS_DIR / "02_Levy" / "BOOST",
    "Rosenbrock": RESULTS_DIR / "03_Rosenbrock" / "BOOST",
}
VARIANT_DIRS = {
    "default": None,
    "random_sampling": ABLATION_RESULTS_DIR / "1_Random_Sampling",
    "ratio_1_1": ABLATION_RESULTS_DIR / "2_Ratio_1_1",
    "ratio_1_4": ABLATION_RESULTS_DIR / "2_Ratio_1_4",
    "ratio_2_1": ABLATION_RESULTS_DIR / "2_Ratio_2_1",
    "ratio_4_1": ABLATION_RESULTS_DIR / "2_Ratio_4_1",
    "percentile_0": ABLATION_RESULTS_DIR / "3_Percentile_0",
    "percentile_10": ABLATION_RESULTS_DIR / "3_Percentile_10",
    "unlimited": ABLATION_RESULTS_DIR / "4_Unlimited_max_iter",
    "random_tie_breaking": ABLATION_RESULTS_DIR / "5_Random_tie_breaking",
    "ei": ABLATION_RESULTS_DIR / "6_EI",
    "matern32": ABLATION_RESULTS_DIR / "6_Matern32",
}

SUPTITLE_SIZE = 32
TITLE_SIZE = 24
LABEL_SIZE = 24
TICK_SIZE = 24
LEGEND_SIZE = 20
ROW_TITLE_SIZE = 24
INITIAL_POINTS = 10
MAX_SAMPLES = 100
PLOT_START_INDEX = INITIAL_POINTS - 1

rcParams["mathtext.default"] = "regular"
rcParams["mathtext.fontset"] = "stix"


def create_plot(ax, x, y, std=None, label=None, color=None, linestyle=None):
    clipped_std = std * 0.3 if std is not None else 0.0
    ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=3.0, alpha=1.0)
    if std is not None:
        lower = np.maximum(y - clipped_std, np.finfo(float).tiny)
        upper = np.maximum(y + clipped_std, np.finfo(float).tiny)
        ax.fill_between(x, lower, upper, color=color, alpha=0.1, clip_on=True)


def resolve_results_path(objective: str, variant_key: str) -> Path:
    if variant_key == "default":
        folder = DEFAULT_OBJECTIVE_DIRS[objective]
    else:
        folder = VARIANT_DIRS[variant_key]
    return folder / f"{objective}_recommended_results.xlsx"


def load_statistics(path: Path, limit_seeds: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if limit_seeds is None:
        df = pd.read_excel(path, sheet_name="statistics", header=None)
        x = df.iloc[0, 1 + PLOT_START_INDEX : 1 + MAX_SAMPLES].to_numpy(dtype=float) + 1
        y = df.iloc[1, 1 + PLOT_START_INDEX : 1 + MAX_SAMPLES].to_numpy(dtype=float)
        std = df.iloc[2, 1 + PLOT_START_INDEX : 1 + MAX_SAMPLES].to_numpy(dtype=float)
        return x, y, std

    df = pd.read_excel(path, sheet_name="combined_seeds", header=None)
    x = df.iloc[0, 1 + PLOT_START_INDEX : 1 + MAX_SAMPLES].to_numpy(dtype=float) + 1
    seed_values = df.iloc[1 : 1 + limit_seeds, 1 + PLOT_START_INDEX : 1 + MAX_SAMPLES].to_numpy(dtype=float)
    y = np.mean(seed_values, axis=0)
    std = np.std(seed_values, axis=0)
    return x, y, std


def plot_multiple_comparisons_one_figure(variant_sets, label_sets, row_titles, objectives, save_path, scale_type="log"):
    plt.rcParams["font.family"] = "DejaVu Serif"
    nrows = len(variant_sets)
    ncols = len(objectives)

    fig, axs = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows), constrained_layout=True)
    axs = np.atleast_2d(axs)

    colors = ["#d62728", "#2ca02c", "#1f77b4", "#ff7f0e", "#17becf", "#bcbd22"]
    linestyles = ["-", ":", "--", "-.", "--", "-."]

    for row in range(nrows):
        variant_set = variant_sets[row]
        label_set = label_sets[row]

        for col in range(ncols):
            objective = objectives[col]
            ax = axs[row, col]
            all_y_data = []

            for i, (variant_key, label) in enumerate(zip(variant_set, label_set)):
                file_path = resolve_results_path(objective, variant_key)
                if not file_path.exists():
                    print(f"[warning] missing file: {file_path}")
                    continue

                limit_seeds = 10 if variant_key == "default" else None
                x, y, std = load_statistics(file_path, limit_seeds=limit_seeds)
                all_y_data.extend(y[np.isfinite(y)])
                create_plot(
                    ax,
                    x,
                    y,
                    std=std,
                    label=label,
                    color=colors[i % len(colors)],
                    linestyle=linestyles[i % len(linestyles)],
                )

            handles, labels = ax.get_legend_handles_labels()
            n_items = len(labels)
            ncol = 1 if n_items <= 4 else 2
            ax.legend(
                loc="upper right",
                fontsize=LEGEND_SIZE,
                frameon=True,
                ncol=ncol,
                columnspacing=0.5,
                handletextpad=0.5,
                handlelength=1.5,
                framealpha=0.9,
            )

            if all_y_data:
                all_y_data = np.asarray(all_y_data, dtype=float)
                positive_y = all_y_data[all_y_data > 0]
                if scale_type == "log" and positive_y.size:
                    y_min_calc = np.min(positive_y) * 0.8
                    lowerbound = 0.0506 if objective == "Ackley" else 1.3938e-4
                    y_min = max(y_min_calc, lowerbound)
                    y_max = np.max(positive_y) * 1.2
                    ax.set_yscale("log")
                    ax.set_ylim(bottom=y_min, top=y_max)
                    ax.yaxis.set_major_formatter(LogFormatterSciNotation())
                else:
                    y_min = float(np.min(all_y_data))
                    y_max = float(np.max(all_y_data))
                    margin = (y_max - y_min) * 0.05 if y_max > y_min else max(abs(y_max), 1.0) * 0.05
                    ax.set_ylim(bottom=y_min - margin, top=y_max + margin)

            if row == 0:
                ax.set_title(objective, fontsize=TITLE_SIZE)
            # if col == 0:
            #     ax.text(
            #         -0.17,
            #         0.5,
            #         row_titles[row],
            #         transform=ax.transAxes,
            #         fontsize=ROW_TITLE_SIZE,
            #         ha="center",
            #         va="center",
            #         rotation=90,
            #         weight="bold",
            #     )
            ax.set_ylabel("Regret", fontsize=LABEL_SIZE)
            ax.set_xlabel("Evaluation", fontsize=LABEL_SIZE)
            ax.set_xlim(INITIAL_POINTS, MAX_SAMPLES)
            ax.tick_params(axis="both", labelsize=TICK_SIZE)
            ax.grid(True, alpha=0.3)

    saved_paths = save_figure(fig, save_path.parent, save_path.stem, dpi=300)
    plt.close(fig)
    for path in saved_paths:
        print(f"saved: {path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fixed_hyperparameter_case = [["default", "matern32", "ei"]]
    fixed_hyperparameter_labels = [["Default", r"$k$ = Matérn 3/2", r"$\alpha$ = EI"]]

    percentile_case = [["default", "percentile_0", "percentile_10"]]
    percentile_labels = [["Default (Percentile 5)", "Percentile 0", "Percentile 10"]]

    kmeans_case = [["default", "random_sampling"]]
    kmeans_labels = [["Default (K-means Clustering)", "Random Sampling"]]

    priority_case = [["default", "random_tie_breaking"]]
    priority_labels = [["Default (Priority Rule)", "Random Selection"]]

    ratio_case = [["default", "ratio_1_1", "ratio_2_1", "ratio_1_4", "ratio_4_1"]]
    ratio_labels = [["Default (1:2)", "1:1", "2:1", "1:4", "4:1"]]

    unlimited_case = [["default", "unlimited"]]
    unlimited_labels = [["Default (max iter 20)", "Unlimited"]]

    plot_multiple_comparisons_one_figure(
        percentile_case,
        percentile_labels,
        ["Stopping criteria"],
        OBJECTIVES,
        OUTPUT_DIR / "ablation_stopping_criteria.png",
    )
    plot_multiple_comparisons_one_figure(
        kmeans_case,
        kmeans_labels,
        ["Partitioning Method"],
        OBJECTIVES,
        OUTPUT_DIR / "ablation_partitioning_method.png",
    )
    plot_multiple_comparisons_one_figure(
        priority_case,
        priority_labels,
        ["Tie-Breaking rule"],
        OBJECTIVES,
        OUTPUT_DIR / "ablation_tie_breaking_rule.png",
    )
    plot_multiple_comparisons_one_figure(
        ratio_case,
        ratio_labels,
        [r"Size of $r_n$"],
        OBJECTIVES,
        OUTPUT_DIR / "ablation_size_of_r_n.png",
    )
    plot_multiple_comparisons_one_figure(
        unlimited_case,
        unlimited_labels,
        ["Size limitation"],
        OBJECTIVES,
        OUTPUT_DIR / "ablation_size_limit_of_r_n.png",
    )
    plot_multiple_comparisons_one_figure(
        fixed_hyperparameter_case,
        fixed_hyperparameter_labels,
        [r"Fixed $k$ or $\alpha$"],
        OBJECTIVES,
        OUTPUT_DIR / "ablation_fixed_kernel_or_acquisition.png",
    )


if __name__ == "__main__":
    main()
