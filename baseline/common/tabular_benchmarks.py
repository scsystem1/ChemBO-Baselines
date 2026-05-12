from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

OCM_NON_VARIABLE_COLUMNS = {"Name", "M1_mol", "M2_mol", "M3_mol"}
DEFAULT_TARGET_CANDIDATES = (
    "yield",
    "Yield",
    "performance",
    "Performance",
    "objective",
    "Objective",
    "target",
    "Target",
)


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    data_path: Path
    target_column: str
    feature_columns: list[str]
    text_columns: list[str]


def _drop_index_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if "" in df.columns:
        drop_cols.append("")
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def _normalize_column_list(
    raw_columns: str | Sequence[str] | None,
) -> list[str] | None:
    if raw_columns is None:
        return None
    if isinstance(raw_columns, str):
        columns = [piece.strip() for piece in raw_columns.split(",") if piece.strip()]
        return columns or None
    columns = [str(piece).strip() for piece in raw_columns if str(piece).strip()]
    return columns or None


def _infer_target_column(df: pd.DataFrame, dataset_name: str) -> str:
    for candidate in DEFAULT_TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Unable to infer target column for dataset '{dataset_name}'. "
        "Please provide --target-column explicitly."
    )


def _resolve_builtin_dataset(
    root: Path,
    dataset_name: str,
) -> tuple[Path | None, str | None, set[str]]:
    dataset_key = dataset_name.strip().lower()
    if dataset_key == "dar":
        return root / "data" / "DAR.csv", "yield", set()
    if dataset_key == "ocm":
        return root / "data" / "OCM.csv", "Performance", set(OCM_NON_VARIABLE_COLUMNS)
    if dataset_key == "scr":
        return root / "data" / "SCR.csv", None, set()
    if dataset_key == "oer":
        return root / "data" / "OER.csv", "objective", set()
    if dataset_key == "suzuki":
        return root / "data" / "suzuki.csv", "Product_Yield_PCT_Area_UV", set()
    auto_path = root / "data" / f"{dataset_name.upper()}.csv"
    if auto_path.exists():
        return auto_path, None, set()
    return None, None, set()


def _validate_columns(df: pd.DataFrame, column_names: Iterable[str], label: str) -> None:
    missing = [column for column in column_names if column not in df.columns]
    if missing:
        raise ValueError(f"Unknown {label} column(s): {missing}. Available columns: {list(df.columns)}")


def load_benchmark_spec(
    root: Path,
    dataset_name: str,
    *,
    data_path: str | Path | None = None,
    target_column: str | None = None,
    feature_columns: str | Sequence[str] | None = None,
    exclude_columns: str | Sequence[str] | None = None,
    text_columns: str | Sequence[str] | None = None,
) -> tuple[BenchmarkSpec, pd.DataFrame]:
    dataset_key = dataset_name.strip().lower()
    builtin_path, builtin_target, builtin_exclude = _resolve_builtin_dataset(root, dataset_name)
    resolved_path = Path(data_path).expanduser().resolve() if data_path is not None else builtin_path
    if resolved_path is None:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Provide --data-path and --target-column "
            "for a custom tabular dataset."
        )
    if not resolved_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {resolved_path}")

    df = pd.read_csv(resolved_path)
    df = _drop_index_like_columns(df)
    resolved_target = target_column or builtin_target or _infer_target_column(df, dataset_name)
    if resolved_target not in df.columns:
        raise ValueError(
            f"Target column '{resolved_target}' not found in dataset {resolved_path}. "
            f"Available columns: {list(df.columns)}"
        )

    requested_feature_columns = _normalize_column_list(feature_columns)
    requested_exclude_columns = set(_normalize_column_list(exclude_columns) or [])
    requested_text_columns = _normalize_column_list(text_columns)
    excluded_columns = set(builtin_exclude) | requested_exclude_columns | {resolved_target}

    if requested_feature_columns is None:
        resolved_feature_columns = [
            column for column in df.columns if column not in excluded_columns
        ]
    else:
        _validate_columns(df, requested_feature_columns, "feature")
        resolved_feature_columns = requested_feature_columns

    if not resolved_feature_columns:
        raise ValueError("Feature column list is empty after resolving dataset configuration.")

    if requested_text_columns is None:
        resolved_text_columns = list(resolved_feature_columns)
    else:
        _validate_columns(df, requested_text_columns, "text")
        resolved_text_columns = requested_text_columns

    spec = BenchmarkSpec(
        name=dataset_key,
        data_path=resolved_path,
        target_column=resolved_target,
        feature_columns=list(resolved_feature_columns),
        text_columns=list(resolved_text_columns),
    )
    return spec, df


def _format_text_value(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return str(value)


def dataset_row_to_text(
    dataset_name: str,
    row: pd.Series,
    text_columns: Sequence[str] | None = None,
) -> str:
    dataset_key = dataset_name.strip().lower()
    if dataset_key == "dar" and {
        "base_SMILES",
        "ligand_SMILES",
        "solvent_SMILES",
        "concentration",
        "temperature",
    }.issubset(set(row.index)):
        return (
            "Direct arylation reaction with "
            f"base {row['base_SMILES']}, "
            f"ligand {row['ligand_SMILES']}, "
            f"solvent {row['solvent_SMILES']}, "
            f"concentration {row['concentration']} M, "
            f"temperature {row['temperature']} C."
        )
    if dataset_key == "ocm" and {
        "Name",
        "M1",
        "M2",
        "M3",
        "Support",
        "M1_mol",
        "M2_mol",
        "M3_mol",
        "Temp",
        "Ar_flow",
        "CH4_flow",
        "O2_flow",
        "CT",
    }.issubset(set(row.index)):
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
    if dataset_key == "suzuki" and {
        "Reactant_1_Name",
        "Reactant_2_Name",
        "Ligand_Short_Hand",
        "Reagent_1_Short_Hand",
        "Solvent_1_Short_Hand",
    }.issubset(set(row.index)):
        return (
            "Suzuki-Miyaura cross-coupling reaction with "
            f"reactant 1 {row['Reactant_1_Name']}, "
            f"reactant 2 {row['Reactant_2_Name']}, "
            f"ligand {row['Ligand_Short_Hand']}, "
            f"base/reagent {row['Reagent_1_Short_Hand']}, "
            f"solvent {row['Solvent_1_Short_Hand']}."
        )
    if dataset_key == "oer" and {
        "ni_load",
        "fe_load",
        "co_load",
        "mn_load",
        "ce_load",
        "la_load",
    }.issubset(set(row.index)):
        return (
            "Oxygen evolution reaction catalyst composition with "
            f"Ni load {row['ni_load']}, "
            f"Fe load {row['fe_load']}, "
            f"Co load {row['co_load']}, "
            f"Mn load {row['mn_load']}, "
            f"Ce load {row['ce_load']}, "
            f"La load {row['la_load']}. "
            "The optimization goal is to maximize objective."
        )

    resolved_text_columns = list(text_columns or row.index.tolist())
    text_parts = [
        f"{column}={_format_text_value(row[column])}"
        for column in resolved_text_columns
        if column in row.index
    ]
    dataset_label = dataset_name.strip().upper()
    return f"{dataset_label} experiment with " + ", ".join(text_parts) + "."


def dataframe_to_texts(
    dataset_name: str,
    df: pd.DataFrame,
    text_columns: Sequence[str] | None = None,
) -> list[str]:
    return [
        dataset_row_to_text(dataset_name, row, text_columns=text_columns)
        for _, row in df.iterrows()
    ]


def dataframe_to_one_hot(df: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    feature_df = df.loc[:, list(feature_columns)].copy()
    categorical_cols = [
        col for col in feature_df.columns if feature_df[col].dtype == "object"
    ]
    if categorical_cols:
        feature_df = pd.get_dummies(feature_df, columns=categorical_cols)
    return feature_df
