from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.core.metric import Metric, MetricFetchE, MetricFetchResult
from ax.core.types import TParameterization, TParamValue
from ax.utils.common.result import Err, Ok
from pyre_extensions import none_throws


def _drop_index_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if "" in df.columns:
        drop_cols.append("")
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


@dataclass(frozen=True)
class TabularChemistryData:
    param_names: list[str]
    objective_dict: dict[tuple[TParamValue, ...], float]

    def evaluate(self, params: TParameterization) -> float:
        key = tuple(params[pname] for pname in self.param_names)
        return self.objective_dict[key]


def load_tabular_chemistry_data(
    data_path: str | Path,
    target_column: str,
    feature_columns: list[str] | None = None,
) -> TabularChemistryData:
    df = pd.read_csv(data_path)
    df = _drop_index_like_columns(df)
    param_names = feature_columns or [col for col in df.columns if col != target_column]
    return TabularChemistryData(
        param_names=param_names,
        objective_dict=df.set_index(param_names)[target_column].to_dict(),
    )


class TabularChemistryMetric(Metric):
    def __init__(
        self,
        name: str,
        data_path: str | Path,
        target_column: str,
        feature_columns: list[str] | None = None,
        noiseless: bool = True,
        lower_is_better: bool = False,
    ) -> None:
        self.noiseless = noiseless
        self.data_path = str(data_path)
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.data = load_tabular_chemistry_data(
            data_path, target_column, feature_columns=feature_columns
        )
        super().__init__(name=name, lower_is_better=lower_is_better)

    def clone(self) -> "TabularChemistryMetric":
        return self.__class__(
            name=self._name,
            data_path=self.data_path,
            target_column=self.target_column,
            feature_columns=self.feature_columns,
            noiseless=self.noiseless,
            lower_is_better=none_throws(self.lower_is_better),
        )

    def fetch_trial_data(self, trial: BaseTrial, **kwargs: Any) -> MetricFetchResult:
        try:
            noise_sd = 0.0 if self.noiseless else float("nan")
            arm_names = []
            means = []
            sem = []
            trial_indices = []

            for name, arm in trial.arms_by_name.items():
                try:
                    val = self.data.evaluate(params=arm.parameters)
                except KeyError as exc:
                    return Err(
                        MetricFetchE(
                            message=f"Arm parameters not found in tabular dataset for {self.name}",
                            exception=exc,
                        )
                    )
                arm_names.append(name)
                means.append(val)
                sem.append(noise_sd)
                trial_indices.append(trial.index)

            df = pd.DataFrame(
                {
                    "arm_name": arm_names,
                    "metric_name": self.name,
                    "mean": means,
                    "sem": sem,
                    "trial_index": trial_indices,
                }
            )
            return Ok(value=Data(df=df))
        except Exception as exc:
            return Err(
                MetricFetchE(
                    message=f"Failed to fetch {self.name}",
                    exception=exc,
                )
            )
