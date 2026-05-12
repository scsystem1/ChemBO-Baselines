from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class BaselineEntry:
    name: str
    conda_env: str
    entrypoint: Path
    default_init_size: int | None
    supported_datasets: tuple[str, ...] | None = None
    supports_custom_data: bool = True


BASELINE_REGISTRY = {
    "bora": BaselineEntry(
        name="bora",
        conda_env="bora",
        entrypoint=ROOT / "baseline" / "bora" / "run_tabular_bora.py",
        default_init_size=5,
    ),
    "cake": BaselineEntry(
        name="cake",
        conda_env="cake",
        entrypoint=ROOT / "baseline" / "cake" / "run_tabular_cake.py",
        default_init_size=5,
    ),
    "gollum": BaselineEntry(
        name="gollum",
        conda_env="gollum",
        entrypoint=ROOT / "baseline" / "gollum" / "run_tabular_gollum.py",
        default_init_size=10,
    ),
    "hebo": BaselineEntry(
        name="hebo",
        conda_env="LLMBO_env",
        entrypoint=ROOT / "baseline" / "hebo" / "run_tabular_hebo.py",
        default_init_size=5,
    ),
    "lmabo": BaselineEntry(
        name="lmabo",
        conda_env="lmabo",
        entrypoint=ROOT / "baseline" / "lmabo" / "run_tabular_lmabo.py",
        default_init_size=5,
    ),
    "pref_bo": BaselineEntry(
        name="pref_bo",
        conda_env="prefbo",
        entrypoint=ROOT / "baseline" / "Pref-BO" / "run_tabular_preference_bo.py",
        default_init_size=1,
        supported_datasets=("dar", "ocm"),
        supports_custom_data=False,
    ),
    "reasoning_bo": BaselineEntry(
        name="reasoning_bo",
        conda_env="reasoning_bo",
        entrypoint=ROOT / "baseline" / "Reasoning-BO" / "run_tabular_reasoning_bo.py",
        default_init_size=3,
    ),
}

BASELINE_ALIASES = {
    "bora": "bora",
    "cake": "cake",
    "gollum": "gollum",
    "hebo": "hebo",
    "lmabo": "lmabo",
    "pref-bo": "pref_bo",
    "pref_bo": "pref_bo",
    "prefbo": "pref_bo",
    "preference-bo": "pref_bo",
    "preference_bo": "pref_bo",
    "reasoning_bo": "reasoning_bo",
    "reasoningbo": "reasoning_bo",
    "reasoning-bo": "reasoning_bo",
}


def log(message: str) -> None:
    print(message, flush=True)


def normalize_baseline_name(raw_name: str) -> str:
    normalized = BASELINE_ALIASES.get(raw_name.strip().lower())
    if normalized is None:
        supported = ", ".join(sorted(BASELINE_REGISTRY))
        raise ValueError(f"Unsupported baseline '{raw_name}'. Supported baselines: {supported}")
    return normalized


def sanitize_name(raw_name: str) -> str:
    text = raw_name.strip()
    sanitized = []
    for char in text:
        if char.isalnum() or char in {"-", "_", "."}:
            sanitized.append(char)
        else:
            sanitized.append("-")
    result = "".join(sanitized).strip("-")
    return result or "run"


def validate_baseline_dataset_selection(args, baseline_entry: BaselineEntry) -> None:
    dataset_key = args.dataset.strip().lower()
    if baseline_entry.supported_datasets is not None and dataset_key not in baseline_entry.supported_datasets:
        supported = ", ".join(baseline_entry.supported_datasets)
        raise ValueError(
            f"Baseline '{baseline_entry.name}' only supports datasets: {supported}. "
            f"Received --dataset={args.dataset!r}."
        )
    if args.data_path and not baseline_entry.supports_custom_data:
        raise ValueError(
            f"Baseline '{baseline_entry.name}' does not currently support --data-path custom datasets."
        )


def build_run_name(
    baseline: str,
    dataset: str,
    trials: int,
    total_budget: int,
    init_size: int | None,
    seed_start: int,
    trial_start_index: int,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    init_text = f"init{init_size}" if init_size is not None else "initauto"
    return (
        f"{timestamp}__{baseline}__{sanitize_name(dataset)}__"
        f"trials{trials}__budget{total_budget}__{init_text}__"
        f"seed{seed_start}__trial{trial_start_index}"
    )


def append_optional_arg(command: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    command.extend([flag, str(value)])


def build_output_dir(args, baseline: str, init_size: int | None) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    run_name = sanitize_name(
        args.run_name
        or build_run_name(
            baseline=baseline,
            dataset=args.dataset,
            trials=args.trials,
            total_budget=args.total_budget,
            init_size=init_size,
            seed_start=args.seed_start,
            trial_start_index=args.trial_start_index,
        )
    )
    return output_root / baseline / sanitize_name(args.dataset) / run_name


def build_command(
    args,
    baseline_entry: BaselineEntry,
    output_dir: Path,
    passthrough_args: list[str],
    init_size: int | None,
) -> list[str]:
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        baseline_entry.conda_env,
        "python",
        str(baseline_entry.entrypoint),
        "--dataset",
        args.dataset,
        "--trials",
        str(args.trials),
        "--trial-start-index",
        str(args.trial_start_index),
        "--seed-start",
        str(args.seed_start),
        "--total-budget",
        str(args.total_budget),
        "--output-dir",
        str(output_dir),
    ]
    append_optional_arg(command, "--init-size", init_size)
    append_optional_arg(command, "--data-path", args.data_path)
    append_optional_arg(command, "--target-column", args.target_column)
    append_optional_arg(command, "--feature-columns", args.feature_columns)
    append_optional_arg(command, "--exclude-columns", args.exclude_columns)
    append_optional_arg(command, "--text-columns", args.text_columns)
    if args.append_to_existing:
        command.append("--append-to-existing")
    command.extend(passthrough_args)
    return command


def write_manifest(
    manifest_path: Path,
    runner_args,
    baseline_entry: BaselineEntry,
    output_dir: Path,
    command: list[str],
    passthrough_args: list[str],
    init_size: int | None,
) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "baseline": baseline_entry.name,
        "conda_env": baseline_entry.conda_env,
        "entrypoint": str(baseline_entry.entrypoint),
        "output_dir": str(output_dir),
        "dataset": runner_args.dataset,
        "trials": runner_args.trials,
        "trial_start_index": runner_args.trial_start_index,
        "seed_start": runner_args.seed_start,
        "total_budget": runner_args.total_budget,
        "init_size": init_size,
        "data_path": runner_args.data_path,
        "target_column": runner_args.target_column,
        "feature_columns": runner_args.feature_columns,
        "exclude_columns": runner_args.exclude_columns,
        "text_columns": runner_args.text_columns,
        "append_to_existing": runner_args.append_to_existing,
        "passthrough_args": passthrough_args,
        "command": command,
        "command_shell": shlex.join(command),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Unified runner for tabular chemistry baselines. "
            "Unknown arguments are forwarded to the selected baseline entrypoint."
        )
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--trial-start-index", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--total-budget", type=int, default=40)
    parser.add_argument("--init-size", type=int, default=10)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--feature-columns", default=None)
    parser.add_argument("--exclude-columns", default=None)
    parser.add_argument("--text-columns", default=None)
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "baseline_runs"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--append-to-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, passthrough_args = parser.parse_known_args()

    baseline_name = normalize_baseline_name(args.baseline)
    baseline_entry = BASELINE_REGISTRY[baseline_name]
    if not baseline_entry.entrypoint.exists():
        raise FileNotFoundError(
            f"Baseline entrypoint does not exist for '{baseline_name}': {baseline_entry.entrypoint}"
        )
    validate_baseline_dataset_selection(args, baseline_entry)
    resolved_init_size = (
        args.init_size if args.init_size is not None else baseline_entry.default_init_size
    )

    if args.trials <= 0:
        raise ValueError("--trials must be positive.")
    if args.trial_start_index <= 0:
        raise ValueError("--trial-start-index must be positive.")
    if args.seed_start < 0:
        raise ValueError("--seed-start must be non-negative.")
    if args.total_budget <= 0:
        raise ValueError("--total-budget must be positive.")
    if resolved_init_size is not None and resolved_init_size <= 0:
        raise ValueError("--init-size must be positive when provided.")
    if (
        resolved_init_size is not None
        and resolved_init_size >= args.total_budget
        and baseline_name != "reasoning_bo"
    ):
        raise ValueError("--init-size must be smaller than --total-budget.")
    if (
        resolved_init_size is not None
        and resolved_init_size > args.total_budget
        and baseline_name == "reasoning_bo"
    ):
        raise ValueError("--init-size must be smaller than or equal to --total-budget.")

    output_dir = build_output_dir(args, baseline_name, resolved_init_size)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.append_to_existing:
        raise ValueError(
            f"Refusing to reuse non-empty output directory without --append-to-existing: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(args, baseline_entry, output_dir, passthrough_args, resolved_init_size)
    write_manifest(
        manifest_path=output_dir / "run_manifest.json",
        runner_args=args,
        baseline_entry=baseline_entry,
        output_dir=output_dir,
        command=command,
        passthrough_args=passthrough_args,
        init_size=resolved_init_size,
    )

    log(f"[Runner] Baseline={baseline_entry.name} dataset={args.dataset} output_dir={output_dir}")
    log(f"[Runner] Command: {shlex.join(command)}")
    if args.dry_run:
        sys.exit(0)

    completed = subprocess.run(command, cwd=str(ROOT), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
