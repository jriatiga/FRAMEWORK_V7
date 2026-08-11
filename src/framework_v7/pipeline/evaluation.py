"""Evaluation helpers extracted from notebook C15."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from framework_v7.paths import EVALUATIONS_DIR

from .utils import artifact_inventory, discover_experiments, read_key_value_table, read_table


TRANSFORMATIONS_RAW_BASE_URL = (
    "https://raw.githubusercontent.com/jriatiga/FRAMEWORK_V7/refs/heads/main/"
    "DATA/MACHINE_LEARNING/C13_MACHINE_LEARNING/Transformaciones"
)


def experiment_candidates(experiment: str) -> list[str]:
    """Return fallback experiment identifiers for versioned artifacts.

    Args:
        experiment (str): Experiment identifier, for example ``Exp01-V3``.

    Returns:
        list[str]: Ordered candidates, starting with the requested experiment
        and then the base experiment when a version suffix exists.
    """

    candidates = [experiment]
    if "-" in experiment:
        base_experiment = experiment.split("-")[0]
        if base_experiment not in candidates:
            candidates.append(base_experiment)
    return candidates


def transformation_artifact_urls(
    experiment: str,
    artifact_name: str,
    base_url: str = TRANSFORMATIONS_RAW_BASE_URL,
) -> list[str]:
    """Build candidate raw GitHub URLs for a transformation artifact.

    Args:
        experiment (str): Experiment identifier.
        artifact_name (str): Artifact file name, such as
            ``dataset_machine_learning_transformado.parquet`` or
            ``scaler.pkl``.
        base_url (str): Raw GitHub URL for the transformations directory.

    Returns:
        list[str]: Ordered candidate URLs following the C15 fallback strategy.
    """

    root = base_url.rstrip("/")
    return [f"{root}/{candidate}/{artifact_name}" for candidate in experiment_candidates(experiment)]


def transformed_dataset_urls(experiment: str) -> list[str]:
    """Build candidate URLs for the transformed ML dataset.

    Args:
        experiment (str): Experiment identifier.

    Returns:
        list[str]: Candidate raw GitHub URLs.
    """

    return transformation_artifact_urls(experiment, "dataset_machine_learning_transformado.parquet")


def scaler_urls(experiment: str) -> list[str]:
    """Build candidate URLs for the fitted scaler artifact.

    Args:
        experiment (str): Experiment identifier.

    Returns:
        list[str]: Candidate raw GitHub URLs.
    """

    return transformation_artifact_urls(experiment, "scaler.pkl")


def classification_metrics(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> pd.DataFrame:
    """Compute binary-classification metrics without external dependencies.

    The function intentionally avoids scikit-learn so evaluation notebooks can
    run with the lightweight dependency set defined by the repository.

    Args:
        y_true (list[int] | np.ndarray): True binary labels encoded as 0 or 1.
        y_pred (list[int] | np.ndarray): Predicted binary labels encoded as 0
            or 1.

    Returns:
        pd.DataFrame: Metrics table with accuracy, precision, recall, F1,
        specificity, balanced accuracy and confusion-matrix counts.

    Raises:
        ValueError: If ``y_true`` and ``y_pred`` have different lengths.
    """

    true = np.asarray(y_true).astype(int)
    pred = np.asarray(y_pred).astype(int)
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must have the same length")

    tp = int(((true == 1) & (pred == 1)).sum())
    tn = int(((true == 0) & (pred == 0)).sum())
    fp = int(((true == 0) & (pred == 1)).sum())
    fn = int(((true == 1) & (pred == 0)).sum())
    total = max(len(true), 1)
    accuracy = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    balanced_accuracy = (recall + specificity) / 2

    return pd.DataFrame(
        [
            {"Indicador": "Accuracy", "Valor": accuracy},
            {"Indicador": "Precision", "Valor": precision},
            {"Indicador": "Recall", "Valor": recall},
            {"Indicador": "F1 Score", "Valor": f1},
            {"Indicador": "Especificidad", "Valor": specificity},
            {"Indicador": "Accuracy Balanceada", "Valor": balanced_accuracy},
            {"Indicador": "TP", "Valor": tp},
            {"Indicador": "TN", "Valor": tn},
            {"Indicador": "FP", "Valor": fp},
            {"Indicador": "FN", "Valor": fn},
        ]
    )


def regression_metrics(y_true: list[float] | np.ndarray, y_pred: list[float] | np.ndarray) -> pd.DataFrame:
    """Compute common regression metrics without external dependencies.

    Args:
        y_true (list[float] | np.ndarray): True numeric target values.
        y_pred (list[float] | np.ndarray): Predicted numeric values.

    Returns:
        pd.DataFrame: Metrics table with MAE, RMSE and R2.

    Raises:
        ValueError: If ``y_true`` and ``y_pred`` have different lengths.
    """

    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must have the same length")
    residuals = true - pred
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    denominator = np.sum((true - np.mean(true)) ** 2)
    r2 = 1 - np.sum(residuals**2) / denominator if denominator else 0.0
    return pd.DataFrame(
        [
            {"Indicador": "MAE", "Valor": mae},
            {"Indicador": "RMSE", "Valor": rmse},
            {"Indicador": "R2", "Valor": r2},
        ]
    )


def threshold_predictions(predictions: list[float] | np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert probabilities or continuous scores into binary labels.

    Args:
        predictions (list[float] | np.ndarray): Numeric scores or
            probabilities.
        threshold (float): Decision threshold. Values greater than or equal to
            this threshold are mapped to 1.

    Returns:
        np.ndarray: Binary label array.
    """

    return (np.asarray(predictions, dtype=float) >= threshold).astype(int)


def recommendations_from_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Generate modeling recommendations from metric diagnostics.

    The rules encode the first evaluation lessons from Exp01: low recall,
    zero precision or weak balanced accuracy indicate imbalance, threshold or
    feature-selection issues.

    Args:
        metrics (pd.DataFrame): Metrics table with ``Indicador`` and ``Valor``
            columns.

    Returns:
        pd.DataFrame: One-column table named ``Recomendacion``.
    """

    values = dict(zip(metrics["Indicador"], pd.to_numeric(metrics["Valor"], errors="coerce")))
    recommendations = []
    if values.get("Recall", 1) == 0:
        recommendations.append("Aplicar balanceo de clases y ajustar el umbral de decision.")
    if values.get("Precision", 1) == 0:
        recommendations.append("Revisar variables predictoras y comparar con modelos base.")
    if values.get("Accuracy Balanceada", 1) < 0.6:
        recommendations.append("Usar metricas balanceadas para seleccionar el mejor modelo.")
    if not recommendations:
        recommendations.append("Mantener el modelo como linea base y validar con nuevos periodos.")
    return pd.DataFrame({"Recomendacion": recommendations})


def load_prediction_metadata(experiment: str, evaluations_dir: Path | None = None) -> dict[str, str]:
    """Load prediction metadata for one experiment.

    Args:
        experiment (str): Experiment identifier.
        evaluations_dir (Path | None): Optional evaluations artifact root.

    Returns:
        dict[str, str]: Prediction metadata as key-value pairs. Returns an
        empty dictionary when the artifact is missing.
    """

    root = evaluations_dir or EVALUATIONS_DIR
    return read_key_value_table(root / experiment / "metadata_prediccion.csv")


def load_predictions(experiment: str, evaluations_dir: Path | None = None) -> pd.DataFrame:
    """Load prediction outputs for one experiment.

    Args:
        experiment (str): Experiment identifier.
        evaluations_dir (Path | None): Optional evaluations artifact root.

    Returns:
        pd.DataFrame: Prediction table with at least ``Registro`` and
        ``Prediccion`` when available.
    """

    root = evaluations_dir or EVALUATIONS_DIR
    return read_table(root / experiment / "predicciones.csv")


def prediction_distribution(predictions: pd.DataFrame, column: str = "Prediccion") -> pd.DataFrame:
    """Describe prediction values for model-auditing notebooks.

    Args:
        predictions (pd.DataFrame): Prediction output table.
        column (str): Numeric prediction column to summarize.

    Returns:
        pd.DataFrame: One-row table with count, min, max, mean and standard
        deviation. Returns an empty table when the column is unavailable.
    """

    if predictions.empty or column not in predictions.columns:
        return pd.DataFrame(columns=["Conteo", "Minimo", "Maximo", "Media", "Desviacion"])
    values = pd.to_numeric(predictions[column], errors="coerce").dropna()
    return pd.DataFrame(
        [
            {
                "Conteo": len(values),
                "Minimo": values.min(),
                "Maximo": values.max(),
                "Media": values.mean(),
                "Desviacion": values.std(),
            }
        ]
    )


def summarize_evaluation_experiments(evaluations_dir: Path | None = None) -> pd.DataFrame:
    """Summarize prediction artifacts by experiment.

    Args:
        evaluations_dir (Path | None): Optional evaluations artifact root.

    Returns:
        pd.DataFrame: One row per experiment with prediction counts and key
        metadata exported by notebook C15.
    """

    root = evaluations_dir or EVALUATIONS_DIR
    rows = []
    for experiment in discover_experiments(root):
        metadata = load_prediction_metadata(experiment, root)
        predictions = load_predictions(experiment, root)
        rows.append(
            {
                "Experimento": experiment,
                "Variable_Objetivo": metadata.get("Variable Objetivo", ""),
                "Modelo": metadata.get("Modelo", ""),
                "Metodo_Transformacion": metadata.get("Metodo Transformacion", ""),
                "Ventana": metadata.get("Ventana", ""),
                "Predicciones": len(predictions),
                "Archivo_Predicciones": not predictions.empty,
            }
        )
    return pd.DataFrame(rows)


def evaluation_artifact_inventory(experiment: str | None = None) -> pd.DataFrame:
    """Inventory C15 prediction and evaluation artifacts.

    Args:
        experiment (str | None): Optional experiment identifier used to limit
            the recursive scan.

    Returns:
        pd.DataFrame: File inventory for evaluation artifacts.
    """

    return artifact_inventory(EVALUATIONS_DIR, experiment)
