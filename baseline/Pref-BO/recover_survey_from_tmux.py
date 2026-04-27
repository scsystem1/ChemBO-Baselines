from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from run_tabular_preference_bo import (
    ROOT,
    build_prompt,
    estimate_prompt_tokens,
    load_benchmark_spec,
)


QUESTION_RE = re.compile(
    r"Survey question (?P<num>\d+)/(?P<total>\d+): compare idx_a=(?P<idx_a>\d+) vs idx_b=(?P<idx_b>\d+)"
)
ANSWER_RE = re.compile(
    r"Survey question (?P<num>\d+)/(?P<total>\d+) answered: pred_setup=(?P<pred>[AB])"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover PrefBO survey progress from a tmux pane dump.")
    parser.add_argument("--dataset", choices=["dar", "ocm"], required=True)
    parser.add_argument("--tmux-log", required=True)
    parser.add_argument("--questions-file", required=True)
    parser.add_argument("--max-questions", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    _, df = load_benchmark_spec(ROOT, args.dataset)
    context_df = df.head(5).copy()
    questions = np.load(args.questions_file, allow_pickle=False)[: args.max_questions]
    tmux_text = Path(args.tmux_log).read_text(encoding="utf-8")

    answered = {}
    for match in ANSWER_RE.finditer(tmux_text):
        q_num = int(match.group("num"))
        answered[q_num - 1] = match.group("pred")

    rows = []
    for q_idx in sorted(answered):
        idx_a, idx_b = questions[q_idx]
        prompt = build_prompt(args.dataset, df.iloc[int(idx_a)], df.iloc[int(idx_b)], context_df)
        rows.append(
            {
                "question": int(q_idx),
                "idx_a": int(idx_a),
                "idx_b": int(idx_b),
                "pred_setup": answered[q_idx],
                "raw_response": "",
                "prompt_tokens_est": estimate_prompt_tokens(prompt),
                "response_tokens_est": np.nan,
                "reasoning": "",
                "status": "success",
                "attempts_used": np.nan,
                "source": "tmux_recovered",
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    survey_csv = output_dir / "llm_survey.csv"
    recovered_csv = output_dir / "llm_survey_recovered_from_tmux.csv"
    remaining_npy = output_dir / "remaining_questions.npy"

    recovered_df = pd.DataFrame(rows).sort_values("question")
    recovered_df.to_csv(recovered_csv, index=False)
    recovered_df.to_csv(survey_csv, index=False)

    answered_set = set(recovered_df["question"].astype(int).tolist())
    remaining = np.array(
        [questions[q_idx] for q_idx in range(len(questions)) if q_idx not in answered_set],
        dtype=int,
    )
    np.save(remaining_npy, remaining)

    print(
        f"[PrefBO][{args.dataset.upper()}] Recovered {len(recovered_df)} answered questions from tmux "
        f"into {survey_csv}",
        flush=True,
    )
    print(
        f"[PrefBO][{args.dataset.upper()}] Remaining unanswered questions: {len(remaining)} "
        f"(saved to {remaining_npy})",
        flush=True,
    )


if __name__ == "__main__":
    main()
