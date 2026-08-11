"""Machine-learning transformation helpers extracted from notebook C13."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from framework_v7.paths import MACHINE_LEARNING_DIR

from .utils import artifact_inventory, discover_experiments, read_key_value_table, read_table


TRANSFORMATIONS_DIR = MACHINE_LEARNING_DIR / "Transformaciones"
DIAGNOSTIC_DIR = MACHINE_LEARNING_DIR / "Diagnostico"


def preprocessing_recommendation(values: pd.Series, skewness: float) -> str:
    """Recommend a preprocessing action for a numeric variable.

    Args:
        values (pd.Series): Values of the variable being diagnosed.
        skewness (float): Skewness value calculated for the variable.

    Returns:
        str: Human-readable preprocessing recommendation.
    """

    if values.nunique(dropna=True) <= 2:
        return "Variable binaria. No requiere transformacion."
    if abs(skewness) < 0.5:
        return "No requiere transformacion."
    if abs(skewness) < 1:
        return "Evaluar estandarizacion."
    return "Considerar escalamiento robusto."


def coverage_level(value: float) -> str:
    """Classify a percentage of coverage into a qualitative level.

    Args:
        value (float): Coverage percentage.

    Returns:
        str: Coverage label.
    """

    if value >= 90:
        return "Excelente"
    if value >= 70:
        return "Buena"
    if value >= 50:
        return "Aceptable"
    if value >= 20:
        return "Baja"
    return "Insuficiente"


def select_transformation_method(asymmetric_variables: int, variables_with_outliers: int = 0) -> str:
    """Select the transformation family suggested by C13 diagnostics.

    Args:
        asymmetric_variables (int): Count of variables with relevant skewness.
        variables_with_outliers (int): Count of variables with outlier alerts.

    Returns:
        str: Name of the recommended transformation method.
    """

    if asymmetric_variables >= 2 or variables_with_outliers >= 2:
        return "RobustScaler"
    return "StandardScaler"


def fit_minmax(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Fit min-max scaling parameters for selected columns.

    Parameters are stored as a small DataFrame so notebooks can export and
    reuse the exact transformation applied to training data.

    Args:
        df (pd.DataFrame): Source dataset.
        columns (list[str]): Numeric columns to scale.

    Returns:
        pd.DataFrame: Table with ``Variable``, ``Min`` and ``Max`` columns.
    """

    rows = []
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce")
        rows.append({"Variable": column, "Min": values.min(), "Max": values.max()})
    return pd.DataFrame(rows)


def apply_minmax(df: pd.DataFrame, scaler: pd.DataFrame) -> pd.DataFrame:
    """Apply min-max scaling from fitted parameters.

    Args:
        df (pd.DataFrame): Source dataset to transform.
        scaler (pd.DataFrame): Parameter table produced by ``fit_minmax`` with
            ``Variable``, ``Min`` and ``Max`` columns.

    Returns:
        pd.DataFrame: Copy of ``df`` with scaled variables where matching
        parameters are available.
    """

    output = df.copy()
    for row in scaler.to_dict("records"):
        column = row["Variable"]
        if column not in output.columns:
            continue
        span = row["Max"] - row["Min"]
        if pd.isna(span) or span == 0:
            output[column] = 0.0
        else:
            output[column] = (pd.to_numeric(output[column], errors="coerce") - row["Min"]) / span
    return output


def create_temporal_sequences(
    df: pd.DataFrame,
    predictors: list[str],
    target: str,
    window: int = 12,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create temporal tensors for sequence models.

    Rows must already be sorted in temporal order. The function builds rolling
    windows for recurrent or sequence-based models such as LSTM.

    Args:
        df (pd.DataFrame): Ordered modeling dataset.
        predictors (list[str]): Predictor columns used as model features.
        target (str): Target column to forecast.
        window (int): Number of past rows per sequence.
        horizon (int): Forecast horizon measured in rows.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple ``(X, y)`` where ``X`` has shape
        ``samples x window x features`` and ``y`` has one target value per
        sample.

    Raises:
        ValueError: If any predictor or target column is missing.
    """

    missing = [column for column in predictors + [target] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for sequences: {missing}")
    values = df[predictors].to_numpy(dtype=float)
    target_values = df[target].to_numpy(dtype=float)
    x_rows = []
    y_rows = []
    last_start = len(df) - window - horizon + 1
    for start in range(max(last_start, 0)):
        end = start + window
        target_index = end + horizon - 1
        x_rows.append(values[start:end])
        y_rows.append(target_values[target_index])
    return np.asarray(x_rows), np.asarray(y_rows)


def transformation_diagnostic(df: pd.DataFrame) -> pd.DataFrame:
    """Build diagnostics for transformed machine-learning data.

    Args:
        df (pd.DataFrame): Transformed modeling dataset.

    Returns:
        pd.DataFrame: Diagnostic table with dimensions, numeric variable count,
        constant variable count and null count.
    """

    numeric_cols = df.select_dtypes(include="number").columns
    constant_cols = [column for column in numeric_cols if df[column].nunique(dropna=True) <= 1]
    return pd.DataFrame(
        [
            {"Indicador": "filas", "Valor": len(df)},
            {"Indicador": "columnas", "Valor": df.shape[1]},
            {"Indicador": "variables_numericas", "Valor": len(numeric_cols)},
            {"Indicador": "variables_constantes", "Valor": len(constant_cols)},
            {"Indicador": "nulos", "Valor": int(df.isna().sum().sum())},
        ]
    )


def load_transformed_dataset(experiment: str, transformations_dir: Path | None = None) -> pd.DataFrame:
    """Load the transformed dataset for one experiment.

    Args:
        experiment (str): Experiment identifier, for example ``Exp04``.
        transformations_dir (Path | None): Optional root directory containing
            transformation outputs. When omitted, the repository C13 path is
            used.

    Returns:
        pd.DataFrame: Transformed machine-learning dataset. Returns an empty
        DataFrame when the CSV artifact is unavailable.
    """

    root = transformations_dir or TRANSFORMATIONS_DIR
    path = root / experiment / "dataset_machine_learning_transformado.csv"
    return read_table(path)


def load_sequence_metadata(experiment: str, transformations_dir: Path | None = None) -> dict[str, str]:
    """Load sequence-preparation metadata for one experiment.

    Args:
        experiment (str): Experiment identifier.
        transformations_dir (Path | None): Optional C13 transformations root.

    Returns:
        dict[str, str]: Metadata exported by the sequence preparation notebook.
    """

    root = transformations_dir or TRANSFORMATIONS_DIR
    return read_key_value_table(root / experiment / "metadata_secuencias.csv")


def load_preparation_record(experiment: str, transformations_dir: Path | None = None) -> dict[str, str]:
    """Load the C13 preparation record for one experiment.

    Args:
        experiment (str): Experiment identifier.
        transformations_dir (Path | None): Optional C13 transformations root.

    Returns:
        dict[str, str]: Key-value preparation record. Returns an empty
        dictionary when the artifact is missing.
    """

    root = transformations_dir or TRANSFORMATIONS_DIR
    return read_key_value_table(root / experiment / "registro_preparacion.csv")


def load_ml_diagnostic(experiment: str, diagnostic_dir: Path | None = None) -> pd.DataFrame:
    """Load the statistical machine-learning diagnostic for an experiment.

    Args:
        experiment (str): Experiment identifier.
        diagnostic_dir (Path | None): Optional C13 diagnostic root.

    Returns:
        pd.DataFrame: Diagnostic table exported by C13. Returns an empty
        DataFrame when the artifact is absent.
    """

    root = diagnostic_dir or DIAGNOSTIC_DIR
    path = root / experiment / "diagnostico_estadistico_ml.csv"
    fallback = root / "diagnostico_estadistico_ml.csv"
    return read_table(path) if path.exists() else read_table(fallback)


def summarize_ml_experiments(transformations_dir: Path | None = None) -> pd.DataFrame:
    """Summarize transformed datasets and sequence metadata by experiment.

    Args:
        transformations_dir (Path | None): Optional C13 transformations root.

    Returns:
        pd.DataFrame: One row per experiment with availability flags, dataset
        dimensions and selected metadata values.
    """

    root = transformations_dir or TRANSFORMATIONS_DIR
    rows = []
    for experiment in discover_experiments(root):
        dataset = load_transformed_dataset(experiment, root)
        metadata = load_sequence_metadata(experiment, root)
        record = load_preparation_record(experiment, root)
        rows.append(
            {
                "Experimento": experiment,
                "Dataset": not dataset.empty,
                "Filas": len(dataset),
                "Columnas": dataset.shape[1],
                "Variable_Objetivo": metadata.get("Variable Objetivo", record.get("Variable Objetivo", "")),
                "Ventana": metadata.get("Ventana", record.get("Ventana Temporal", "")),
                "Horizonte": metadata.get("Horizonte", record.get("Horizonte", "")),
                "Metodo_Transformacion": record.get(
                    "Metodo de Transformacion",
                    record.get("Metodo de Transformaci\u00f3n", ""),
                ),
            }
        )
    return pd.DataFrame(rows)


def ml_artifact_inventory(experiment: str | None = None) -> pd.DataFrame:
    """Inventory C13 machine-learning artifacts.

    Args:
        experiment (str | None): Optional experiment identifier used to limit
            the recursive scan.

    Returns:
        pd.DataFrame: File inventory for transformation artifacts.
    """

    return artifact_inventory(TRANSFORMATIONS_DIR, experiment)
