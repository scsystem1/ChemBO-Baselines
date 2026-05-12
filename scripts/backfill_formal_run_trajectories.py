from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from baseline.common.experiment_tracking import (
    build_trial_trace,
    export_trajectory_artifacts,
    load_existing_trajectory_artifacts,
)
from baseline.common.tabular_benchmarks import load_benchmark_spec


def iter_trajectory_npz_paths(formal_runs_root: Path) -> list[Path]:
    return sorted(formal_runs_root.glob("*/*/*_trajectory.npz"))


def infer_stem(trajectory_path: Path) -> str:
    suffix = "_trajectory.npz"
    if not trajectory_path.name.endswith(suffix):
        raise ValueError(f"Unexpected trajectory filename: {trajectory_path.name}")
    return trajectory_path.name[: -len(suffix)]


def infer_init_size(trajectory_path: Path, stem: str) -> int | None:
    summary_name = f"{stem}_summary.json"
    summary_path = trajectory_path.parent / summary_name
    if not summary_path.exists():
        return None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    init_size = payload.get("init_size")
    return None if init_size is None else int(init_size)


def with_inferred_phases(traces, init_size: int | None):
    if init_size is None:
        return traces
    rebuilt = []
    for trace in traces:
        phases = [
            "init" if idx < init_size else "bo"
            for idx in range(trace.length)
        ]
        rebuilt.append(
            build_trial_trace(
                selected_indices=np.asarray(trace.selected_indices, dtype=int).tolist(),
                observed_values=np.asarray(trace.observed_values, dtype=float).tolist(),
                phases=phases,
            )
        )
    return rebuilt


def backfill_trajectory(trajectory_path: Path) -> bool:
    payload = load_existing_trajectory_artifacts(trajectory_path)
    if payload is None:
        return False

    traces, trial_numbers = payload
    dataset_name = trajectory_path.parent.name
    stem = infer_stem(trajectory_path)
    init_size = infer_init_size(trajectory_path, stem)
    traces = with_inferred_phases(traces, init_size)
    spec, df = load_benchmark_spec(ROOT, dataset_name)
    export_trajectory_artifacts(
        output_dir=trajectory_path.parent,
        stem=stem,
        traces=traces,
        trial_numbers=trial_numbers,
        dataset_df=df,
        target_column=spec.target_column,
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill formal run trajectory files with original parameter columns."
    )
    parser.add_argument(
        "--formal-runs-root",
        default=str(ROOT / "outputs" / "formal_runs"),
        help="Root directory containing formal run outputs.",
    )
    args = parser.parse_args()

    formal_runs_root = Path(args.formal_runs_root).expanduser().resolve()
    if not formal_runs_root.exists():
        raise FileNotFoundError(f"formal runs directory not found: {formal_runs_root}")

    updated = 0
    skipped = 0
    for trajectory_path in iter_trajectory_npz_paths(formal_runs_root):
        if backfill_trajectory(trajectory_path):
            updated += 1
            print(f"[updated] {trajectory_path}")
        else:
            skipped += 1
            print(f"[skipped] {trajectory_path}")

    print(
        f"Backfill complete. updated={updated} skipped={skipped} root={formal_runs_root}"
    )


if __name__ == "__main__":
    main()
