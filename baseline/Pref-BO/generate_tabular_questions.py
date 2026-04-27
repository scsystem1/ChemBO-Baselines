from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from run_tabular_preference_bo import generate_question_pairs, load_benchmark_spec, ROOT


DEFAULT_TOTAL_QUESTIONS = {"dar": 5000, "ocm": 20000}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate question-pair files for tabular Pref-BO datasets.")
    parser.add_argument("--dataset", choices=["dar", "ocm"], required=True)
    parser.add_argument("--n-questions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    total_questions = args.n_questions or DEFAULT_TOTAL_QUESTIONS[args.dataset]
    _, df = load_benchmark_spec(ROOT, args.dataset)
    questions = generate_question_pairs(len(df), total_questions, seed=args.seed)

    output_path = Path(
        args.output
        or ROOT / "baseline" / "Pref-BO" / "questions" / f"{args.dataset}_questions_seed{args.seed}.npy"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, questions)
    print(
        f"[PrefBO][{args.dataset.upper()}] Saved {len(questions)} questions to {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
