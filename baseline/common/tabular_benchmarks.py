from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

OCM_NON_VARIABLE_COLUMNS = {"Name", "M1_mol", "M2_mol", "M3_mol"}


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    data_path: Path
    target_column: str
    feature_columns: list[str]


def _drop_index_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if "" in df.columns:
        drop_cols.append("")
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def load_benchmark_spec(root: Path, dataset_name: str) -> tuple[BenchmarkSpec, pd.DataFrame]:
    dataset_key = dataset_name.strip().lower()
    if dataset_key == "dar":
        path = root / "data" / "DAR.csv"
        target_column = "yield"
    elif dataset_key == "ocm":
        path = root / "data" / "OCM.csv"
        target_column = "Performance"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    df = pd.read_csv(path)
    df = _drop_index_like_columns(df)
    if dataset_key == "ocm":
        feature_columns = [
            col for col in df.columns if col != target_column and col not in OCM_NON_VARIABLE_COLUMNS
        ]
    else:
        feature_columns = [col for col in df.columns if col != target_column]
    spec = BenchmarkSpec(
        name=dataset_key,
        data_path=path,
        target_column=target_column,
        feature_columns=feature_columns,
    )
    return spec, df


def dataset_row_to_text(dataset_name: str, row: pd.Series) -> str:
    dataset_key = dataset_name.strip().lower()
    if dataset_key == "dar":
        return (
            "Direct arylation reaction with "
            f"base {row['base_SMILES']}, "
            f"ligand {row['ligand_SMILES']}, "
            f"solvent {row['solvent_SMILES']}, "
            f"concentration {row['concentration']} M, "
            f"temperature {row['temperature']} C."
        )
    if dataset_key == "ocm":
        return (
            "Oxidative coupling of methane catalyst and condition with "
            f"catalyst {row['Name']}, "
            f"metals {row['M1']}/{row['M2']}/{row['M3']}, "
            f"support {row['Support']}, "
            f"metal mol fractions {row['M1_mol']}/{row['M2_mol']}/{row['M3_mol']}, "
            f"temperature {row['Temp']} C, "
            f"Ar flow {row['Ar_flow']}, "
            f"CH4 flow {row['CH4_flow']}, "
            f"O2 flow {row['O2_flow']}, "
            f"contact time {row['CT']}."
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def dataframe_to_texts(dataset_name: str, df: pd.DataFrame) -> list[str]:
    return [dataset_row_to_text(dataset_name, row) for _, row in df.iterrows()]


def dataframe_to_one_hot(df: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    feature_df = df.loc[:, list(feature_columns)].copy()
    categorical_cols = [
        col for col in feature_df.columns if feature_df[col].dtype == "object"
    ]
    if categorical_cols:
        feature_df = pd.get_dummies(feature_df, columns=categorical_cols)
    return feature_df
