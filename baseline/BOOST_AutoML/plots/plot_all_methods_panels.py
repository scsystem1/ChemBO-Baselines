from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterSciNotation
from plot_config import METHOD_PANELS_DIR, RESULTS_DIR, save_figure, slugify

from plot_common import (
    SPECIAL_STYLES,
    build_result_bundle,
    build_special_sources,
    compute_regret_floor,
    format_combo_label,
    infer_objective_stem,
    iter_fixed_results,
    parse_objective_name,
    parse_title,
    plot_boxplot,
    plot_quantile,
    plot_remaining_gap,
    plot_with_ci,
    style_regret_axis,
    style_remaining_gap_axis,
    OPTIMUM_THRESHOLD,
)


TARGET_OBJECTIVES = [
    "01_Ackley",
    "02_Levy",
    "03_Rosenbrock",
    "5636_6D",
    "6762_6D",
    "5891_8D",
    "6322_8D",
    "5964_9D",
    "6794_10D",
    "2277_15D",
    "5906_16D",
]
COMPOSITE_DIR = METHOD_PANELS_DIR
SPECIAL_LEGEND_ORDER = [
    "BOOST (Ours)",
    "BOOST (fixed HP)",
    "Random",
    "Random-KA",
    "Static",
    "Adaptive Kernel",
    "Adaptive Acquisition",
    "qLogNEI (q=1)",
    "HEBO",
]
SUPTITLE_SIZE = 32
TITLE_SIZE = 24
LABEL_SIZE = 24
TICK_SIZE = 24
LEGEND_SIZE = 24
BOX_TITLE_SIZE = 20
BOX_LABEL_SIZE = 20
BOX_TICK_SIZE = 16
ROSENBROCK_BOX_BREAK = (1500.0, 5500.0)


def display_label(label: str) -> str:
    if label == "BOOST":
        return "BOOST (Ours)"
    return label


def sort_entries(entries: list[dict]) -> list[dict]:
    boost_entries = [entry for entry in entries if entry["sort_label"] == "BOOST"]
    others = [entry for entry in entries if entry["sort_label"] != "BOOST"]

    def entry_key(entry: dict):
        values = np.asarray(entry["values"], dtype=float)
        median = float(np.median(values))
        q1, q3 = np.percentile(values, [25, 75])
        return (median, float(q3 - q1), entry["label"])

    others.sort(key=entry_key)
    return boost_entries + others


def collect_objective_data(objective_dir: Path) -> dict:
    folder_name = objective_dir.name
    objective_name = parse_objective_name(folder_name)
    objective_stem = infer_objective_stem(objective_dir, objective_name)
    special_sources = build_special_sources(objective_dir, objective_stem)

    fixed_results = []
    for kernel, acquisition, path in iter_fixed_results(objective_dir, objective_stem):
        bundle = build_result_bundle(path)
        fixed_results.append(
            {
                "label": format_combo_label(kernel, acquisition),
                "stats": bundle["stats"],
                "quantile": bundle["quantile"],
                "samples_to_optimum": bundle["samples_to_optimum"],
                "final_regrets": bundle["final_regrets"],
                "gap_closed": bundle["gap_closed"],
            }
        )

    special_results = {label: build_result_bundle(path) for label, path in special_sources.items()}
    fixed_labels = {result["label"] for result in fixed_results}
    all_method_results = list(fixed_results)
    all_method_results.extend(
        {"label": label, **bundle}
        for label, bundle in special_results.items()
        if label not in fixed_labels
    )

    return {
        "folder_name": folder_name,
        "title": parse_title(folder_name, objective_name, objective_stem),
        "all_method_results": all_method_results,
        "use_samples_to_optimum": any(
            np.any(result["stats"]["means"] <= OPTIMUM_THRESHOLD) for result in all_method_results
        ),
    }


def create_axes(fig: plt.Figure, horizontal_gap_bottom: float = 0.07, vertical_gap: float = 0.06) -> list[plt.Axes]:
    left_margin = 0.06
    right_margin = 0.06
    bottom_margin = 0.09
    top_margin = 0.08
    horizontal_gap_top = 0.07

    width = (1.0 - left_margin - right_margin - 2 * horizontal_gap_top) / 3.0
    available_height = 1.0 - bottom_margin - top_margin
    height = (available_height - 3 * vertical_gap) / 4.0

    row_y = []
    cursor_top = 1.0 - top_margin
    for _ in range(4):
        row_y.append(cursor_top - height)
        cursor_top = cursor_top - height - vertical_gap

    first_row_left = 0.5 - (3 * width + 2 * horizontal_gap_top) / 2.0
    bottom_row_left = 0.5 - (2 * width + horizontal_gap_bottom) / 2.0
    three_col_x = [first_row_left + idx * (width + horizontal_gap_top) for idx in range(3)]
    bottom_col_x = [bottom_row_left + idx * (width + horizontal_gap_bottom) for idx in range(2)]

    axes = []
    for row_idx in range(3):
        for x in three_col_x:
            axes.append(fig.add_axes([x, row_y[row_idx], width, height]))
    for x in bottom_col_x:
        axes.append(fig.add_axes([x, row_y[3], width, height]))
    return axes


def clean_log_ticks(ax: plt.Axes, floor: float) -> None:
    ticks = [tick for tick in ax.get_yticks() if tick > floor and np.isfinite(tick)]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(LogFormatterSciNotation())


def enlarge_axis_text(ax: plt.Axes) -> None:
    ax.title.set_fontsize(TITLE_SIZE)
    ax.xaxis.label.set_fontsize(LABEL_SIZE)
    ax.yaxis.label.set_fontsize(LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)


def style_box_axis_text(ax: plt.Axes) -> None:
    ax.title.set_fontsize(BOX_TITLE_SIZE)
    ax.xaxis.label.set_fontsize(BOX_LABEL_SIZE)
    ax.yaxis.label.set_fontsize(BOX_LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=BOX_TICK_SIZE)
    ax.tick_params(axis="x", labelsize=BOX_TICK_SIZE)


def uses_broken_rosenbrock_boxplot(objective_data: dict) -> bool:
    return objective_data["folder_name"] == "03_Rosenbrock" and not objective_data["use_samples_to_optimum"]


def draw_axis_break_marks(ax_top: plt.Axes, ax_bottom: plt.Axes) -> None:
    marker_width = 0.012
    marker_kwargs = {"color": "black", "clip_on": False, "linewidth": 1.4}
    ax_top.plot(
        (-marker_width, +marker_width),
        (-marker_width, +marker_width),
        transform=ax_top.transAxes,
        **marker_kwargs,
    )
    ax_top.plot(
        (1 - marker_width, 1 + marker_width),
        (-marker_width, +marker_width),
        transform=ax_top.transAxes,
        **marker_kwargs,
    )
    ax_bottom.plot(
        (-marker_width, +marker_width),
        (1 - marker_width, 1 + marker_width),
        transform=ax_bottom.transAxes,
        **marker_kwargs,
    )
    ax_bottom.plot(
        (1 - marker_width, 1 + marker_width),
        (1 - marker_width, 1 + marker_width),
        transform=ax_bottom.transAxes,
        **marker_kwargs,
    )


def draw_broken_boxplot(ax: plt.Axes, entries: list[dict], title: str, ylabel: str) -> None:
    fig = ax.figure
    position = ax.get_position()
    ax.set_visible(False)

    top_fraction = 0.26
    gap_fraction = 0.06
    bottom_fraction = 1.0 - top_fraction - gap_fraction

    bottom_height = position.height * bottom_fraction
    gap_height = position.height * gap_fraction
    top_height = position.height * top_fraction

    ax_bottom = fig.add_axes([position.x0, position.y0, position.width, bottom_height])
    ax_top = fig.add_axes(
        [position.x0, position.y0 + bottom_height + gap_height, position.width, top_height],
        sharex=ax_bottom,
    )

    plot_boxplot(ax_bottom, entries, title, ylabel, False)
    plot_boxplot(ax_top, entries, title, ylabel, False)

    lower_max, upper_min = ROSENBROCK_BOX_BREAK
    all_values = np.concatenate([np.asarray(entry["values"], dtype=float) for entry in entries])
    finite_values = all_values[np.isfinite(all_values)]
    upper_max = float(np.max(finite_values)) if finite_values.size else upper_min + 1.0
    upper_max = max(upper_min + 100.0, 1.05 * upper_max)

    ax_bottom.set_ylim(0.0, lower_max)
    ax_top.set_ylim(upper_min, upper_max)

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_top.set_xlabel("")
    ax_top.set_ylabel("")
    ax_bottom.set_title("")

    ax_top.title.set_fontsize(BOX_TITLE_SIZE)
    ax_bottom.yaxis.label.set_fontsize(BOX_LABEL_SIZE)
    ax_bottom.tick_params(axis="x", labelsize=BOX_TICK_SIZE)
    ax_bottom.tick_params(axis="y", labelsize=BOX_TICK_SIZE)
    ax_top.tick_params(axis="y", labelsize=BOX_TICK_SIZE)

    draw_axis_break_marks(ax_top, ax_bottom)


def draw_regret(ax: plt.Axes, objective_data: dict) -> None:
    results = objective_data["all_method_results"]
    floor = compute_regret_floor([result["stats"]["means"] for result in results])
    top = 1.2 * max(result["stats"]["means"][1] for result in results)
    palette = plt.cm.tab20(np.linspace(0, 1, max(len(results), 1)))
    line_idx = 0
    for result in results:
        style = SPECIAL_STYLES.get(result["label"])
        if style is None:
            style = {
                "color": palette[line_idx],
                "linewidth": 1.8,
                "linestyle": ["-", "--", "-.", ":"][line_idx % 4],
                "zorder": 1,
            }
            line_idx += 1
        plot_with_ci(ax, result["stats"], result["label"], floor=floor, **style)
        line = ax.lines[-1]
        line.set_label(display_label(result["label"]))
    style_regret_axis(ax, objective_data["title"], floor, top)
    clean_log_ticks(ax, floor)
    enlarge_axis_text(ax)


def draw_quantile(ax: plt.Axes, objective_data: dict) -> None:
    results = objective_data["all_method_results"]
    floor = compute_regret_floor([result["quantile"]["quantile"] for result in results])
    top = 1.2 * max(result["quantile"]["quantile"][1] for result in results)
    palette = plt.cm.tab20(np.linspace(0, 1, max(len(results), 1)))
    line_idx = 0
    for result in results:
        style = SPECIAL_STYLES.get(result["label"])
        if style is None:
            style = {
                "color": palette[line_idx],
                "linewidth": 1.8,
                "linestyle": ["-", "--", "-.", ":"][line_idx % 4],
                "zorder": 1,
            }
            line_idx += 1
        plot_quantile(ax, result["quantile"], result["label"], floor=floor, **style)
        line = ax.lines[-1]
        line.set_label(display_label(result["label"]))
    style_regret_axis(ax, objective_data["title"], floor, top)
    clean_log_ticks(ax, floor)
    enlarge_axis_text(ax)


def draw_boxplot(ax: plt.Axes, objective_data: dict) -> None:
    results = objective_data["all_method_results"]
    palette = plt.cm.tab20(np.linspace(0, 1, max(len(results), 1)))
    entries = []
    line_idx = 0
    use_samples = objective_data["use_samples_to_optimum"]
    for result in results:
        style = SPECIAL_STYLES.get(result["label"])
        color = style["color"] if style is not None else palette[line_idx]
        if style is None:
            line_idx += 1
        entries.append(
            {
                "label": display_label(result["label"]),
                "sort_label": result["label"],
                "values": result["samples_to_optimum"] if use_samples else result["final_regrets"],
                "color": color,
            }
        )
    entries = sort_entries(entries)
    ylabel = "Evaluations to Optimum" if use_samples else "Regret at Final Evaluation"
    if uses_broken_rosenbrock_boxplot(objective_data):
        draw_broken_boxplot(ax, entries, objective_data["title"], ylabel)
    else:
        plot_boxplot(ax, entries, objective_data["title"], ylabel, False)
        style_box_axis_text(ax)


def draw_gap(ax: plt.Axes, objective_data: dict) -> None:
    results = objective_data["all_method_results"]
    floor = compute_regret_floor([1.0 - result["gap_closed"]["means"] for result in results])
    top = 1.2 * max(1.0 - result["gap_closed"]["means"][1] for result in results)
    palette = plt.cm.tab20(np.linspace(0, 1, max(len(results), 1)))
    line_idx = 0
    for result in results:
        style = SPECIAL_STYLES.get(result["label"])
        if style is None:
            style = {
                "color": palette[line_idx],
                "linewidth": 1.8,
                "linestyle": ["-", "--", "-.", ":"][line_idx % 4],
                "zorder": 1,
            }
            line_idx += 1
        plot_remaining_gap(ax, result["gap_closed"], result["label"], floor=floor, **style)
        line = ax.lines[-1]
        line.set_label(display_label(result["label"]))
    style_remaining_gap_axis(ax, objective_data["title"], floor, top)
    clean_log_ticks(ax, floor)
    enlarge_axis_text(ax)


def build_metric_figure(metric_name: str, objective_data_list: list[dict], drawer) -> list[Path]:
    plt.rcParams["font.family"] = "DejaVu Serif"
    figure_height = 30 if metric_name == "Final Performance" else 27
    fig = plt.figure(figsize=(24, figure_height))
    box_gap = 0.07 if metric_name == "Final Performance" else 0.07  #####
    v_gap = 0.08 if metric_name == "Final Performance" else 0.06
    axes = create_axes(fig, horizontal_gap_bottom=box_gap, vertical_gap=v_gap)

    for ax, objective_data in zip(axes, objective_data_list):
        drawer(ax, objective_data)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        special_rank = {label: idx for idx, label in enumerate(SPECIAL_LEGEND_ORDER)}
        ordered = sorted(
            zip(handles, labels),
            key=lambda item: (
                0 if item[1] in special_rank else 1,
                special_rank.get(item[1], 999),
                item[1],
            ),
        )
        fig.legend(
            [handle for handle, _ in ordered],
            [label for _, label in ordered],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02), ####
            ncol=7,
            fontsize=LEGEND_SIZE,
            frameon=True,
        )
    # fig.suptitle(metric_name, fontsize=SUPTITLE_SIZE, fontweight="bold")
    output_stem = slugify(metric_name.replace('%', 'pct'))
    saved_paths = save_figure(fig, COMPOSITE_DIR, output_stem, dpi=300)
    plt.close(fig)
    return saved_paths


def main() -> None:
    COMPOSITE_DIR.mkdir(parents=True, exist_ok=True)
    objective_data_list = [collect_objective_data(RESULTS_DIR / name) for name in TARGET_OBJECTIVES]

    outputs = [
        build_metric_figure("Regret", objective_data_list, draw_regret),
        build_metric_figure("Worst 10% Regret", objective_data_list, draw_quantile),
        build_metric_figure("Final Performance", objective_data_list, draw_boxplot),
        build_metric_figure("Remaining Gap", objective_data_list, draw_gap),
    ]

    for output_group in outputs:
        for output in output_group:
            print(f"saved: {output}")


if __name__ == "__main__":
    main()
