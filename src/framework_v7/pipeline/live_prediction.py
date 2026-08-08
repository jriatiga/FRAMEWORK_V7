"""Lightweight live prediction helpers for the Streamlit simulator.

The production notebooks keep the deep-learning experiments as persisted
artifacts. This module provides a small, dependency-free baseline that can be
trained inside Streamlit Cloud with ``numpy`` and ``pandas`` only, making it
useful for interactive scenario exploration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


IRCA_TARGET_LABEL = "IRCA / riesgo de calidad del agua"
VOLUME_TARGET_LABEL = "Volumen util diario de masa hidrica"

IRCA_FEATURES = [
    "Precipitacion_mm",
    "Temp_Media_C",
    "Humedad_Relativa",
    "Velocidad_Viento",
    "Radiacion_Solar",
    "ONI",
    "VolumenUtilDiarioMasa",
    "INDICE DE DESEMPENO INSTITUCIONAL",
    "ACCESO A AGUA POTABLE ADECUADO",
    "PORCENTAJE DE LA POBLACION CON ACCESO A METODOS DE SANEAMIENTO ADECUADOS",
    "COBERTURA DE ACUEDUCTO URBANO",
    "COBERTURA DE ALCANTARILLADO URBANO",
    "CONTINUIDAD DE ACUEDUCTO URBANO",
    "AGUAS RESIDUALES TRATADAS",
    "CONDUCTIVIDAD ELECTRICA",
    "DEMANDA BIOQUIMICA DE OXIGENO (DBO5)",
    "DEMANDA QUIMICA DE OXIGENO (DQO)",
    "FOSFORO TOTAL",
    "NITROGENO TOTAL",
    "OXIGENO DISUELTO (OD)",
    "SOLIDOS SUSPENDIDOS TOTALES",
    "TURBIDEZ",
    "pH",
]

VOLUME_FEATURES = [
    "Precipitacion_mm",
    "Temp_Max_C",
    "Temp_Min_C",
    "Temp_Media_C",
    "Humedad_Relativa",
    "Velocidad_Viento",
    "Radiacion_Solar",
    "ONI",
    "Nivel_Minimo",
    "POBLACION TOTAL",
    "DENSIDAD POBLACIONAL",
    "INDICE DE DESEMPENO INSTITUCIONAL",
    "ACCESO A AGUA POTABLE ADECUADO",
    "PORCENTAJE DE LA POBLACION CON ACCESO A METODOS DE SANEAMIENTO ADECUADOS",
    "COBERTURA DE ACUEDUCTO URBANO",
    "COBERTURA DE ACUEDUCTO RURAL",
    "COBERTURA DE ALCANTARILLADO URBANO",
    "COBERTURA DE ALCANTARILLADO RURAL",
    "CONTINUIDAD DE ACUEDUCTO URBANO",
    "AGUAS RESIDUALES TRATADAS",
]

FEATURE_GROUPS = {
    "Clima y variabilidad": [
        "Precipitacion_mm",
        "Temp_Max_C",
        "Temp_Min_C",
        "Temp_Media_C",
        "Humedad_Relativa",
        "Velocidad_Viento",
        "Radiacion_Solar",
        "ONI",
    ],
    "Hidrologia e hidraulica": [
        "VolumenUtilDiarioMasa",
        "Nivel_Minimo",
    ],
    "Gobernanza y servicios": [
        "POBLACION TOTAL",
        "DENSIDAD POBLACIONAL",
        "INDICE DE DESEMPENO INSTITUCIONAL",
        "ACCESO A AGUA POTABLE ADECUADO",
        "PORCENTAJE DE LA POBLACION CON ACCESO A METODOS DE SANEAMIENTO ADECUADOS",
        "COBERTURA DE ACUEDUCTO URBANO",
        "COBERTURA DE ACUEDUCTO RURAL",
        "COBERTURA DE ALCANTARILLADO URBANO",
        "COBERTURA DE ALCANTARILLADO RURAL",
        "CONTINUIDAD DE ACUEDUCTO URBANO",
        "AGUAS RESIDUALES TRATADAS",
    ],
    "Calidad y percepcion": [
        "CONDUCTIVIDAD ELECTRICA",
        "DEMANDA BIOQUIMICA DE OXIGENO (DBO5)",
        "DEMANDA QUIMICA DE OXIGENO (DQO)",
        "FOSFORO TOTAL",
        "NITROGENO TOTAL",
        "OXIGENO DISUELTO (OD)",
        "SOLIDOS SUSPENDIDOS TOTALES",
        "TURBIDEZ",
        "pH",
    ],
}


@dataclass(frozen=True)
class LiveTarget:
    """Configuration for one live prediction target.

    Attributes:
        label: Human-readable target name displayed in the app.
        column: Dataset column used as the response variable.
        features: Candidate predictor columns for the baseline model.
        unit: Unit displayed next to predictions.
        description: Short methodological explanation for the app.
    """

    label: str
    column: str
    features: list[str]
    unit: str
    description: str


@dataclass(frozen=True)
class LivePredictionModel:
    """Fitted ridge-regression baseline for interactive predictions.

    Attributes:
        target: Target configuration used for training.
        features: Predictor columns retained after quality checks.
        coefficients: Ridge coefficients in standardized feature space.
        intercept: Model intercept.
        means: Training means used for standardization.
        stds: Training standard deviations used for standardization.
        defaults: Median values used when an input is missing.
        metrics: Validation and training diagnostics.
    """

    target: LiveTarget
    features: list[str]
    coefficients: np.ndarray
    intercept: float
    means: pd.Series
    stds: pd.Series
    defaults: pd.Series
    metrics: dict[str, float]


LIVE_TARGETS = {
    IRCA_TARGET_LABEL: LiveTarget(
        label=IRCA_TARGET_LABEL,
        column="irca",
        features=IRCA_FEATURES,
        unit="IRCA",
        description=(
            "Baseline para estimar riesgo de calidad del agua combinando clima, "
            "calidad fisicoquimica, disponibilidad y condiciones institucionales."
        ),
    ),
    VOLUME_TARGET_LABEL: LiveTarget(
        label=VOLUME_TARGET_LABEL,
        column="VolumenUtilDiarioMasa",
        features=VOLUME_FEATURES,
        unit="m3",
        description=(
            "Baseline para explorar disponibilidad hidrica a partir de clima, "
            "variabilidad ONI y variables de gobernanza territorial."
        ),
    ),
}


def available_live_targets(df: pd.DataFrame) -> list[str]:
    """Return target labels that can be trained with the given dataset.

    Args:
        df: Dataset maestro or compatible modeling table.

    Returns:
        Labels for targets with a numeric response column and enough rows.
    """

    labels = []
    for label, target in LIVE_TARGETS.items():
        if target.column not in df.columns:
            continue
        values = pd.to_numeric(df[target.column], errors="coerce")
        if values.notna().sum() >= 30:
            labels.append(label)
    return labels


def feature_group(feature: str) -> str:
    """Return the systemic layer associated with a feature.

    Args:
        feature: Predictor column name.

    Returns:
        Layer label used to organize Streamlit controls.
    """

    for group, features in FEATURE_GROUPS.items():
        if feature in features:
            return group
    return "Otras variables"


def feature_profile(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Build descriptive ranges for live input controls.

    Args:
        df: Dataset containing the predictor columns.
        features: Predictor columns retained by the fitted model.

    Returns:
        DataFrame with min, quartiles, median, mean and max per feature.
    """

    rows = []
    for feature in features:
        values = pd.to_numeric(df[feature], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "Variable": feature,
                "Capa": feature_group(feature),
                "Min": float(values.min()),
                "Q25": float(values.quantile(0.25)),
                "Mediana": float(values.median()),
                "Media": float(values.mean()),
                "Q75": float(values.quantile(0.75)),
                "Max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def fit_live_ridge_model(
    df: pd.DataFrame,
    target_label: str,
    alpha: float = 1.0,
    train_fraction: float = 0.8,
) -> LivePredictionModel:
    """Fit a lightweight ridge-regression model for live simulation.

    Args:
        df: Dataset maestro or compatible modeling table.
        target_label: Key from ``LIVE_TARGETS``.
        alpha: Ridge regularization strength.
        train_fraction: Fraction of ordered rows used for training.

    Returns:
        LivePredictionModel with coefficients, defaults and validation metrics.

    Raises:
        ValueError: If the target is unknown or insufficient data is available.
    """

    if target_label not in LIVE_TARGETS:
        raise ValueError(f"Unknown live target: {target_label}")

    target = LIVE_TARGETS[target_label]
    features = [column for column in target.features if column in df.columns]
    if not features:
        raise ValueError("No predictor columns are available for live training.")

    columns = [target.column, *features]
    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(subset=[target.column]).reset_index(drop=True)
    if len(numeric) < 30:
        raise ValueError("Not enough rows are available for live training.")

    x_values = numeric[features].copy()
    defaults = x_values.median(numeric_only=True)
    x_values = x_values.fillna(defaults)
    valid_features = [
        column
        for column in features
        if x_values[column].notna().all() and float(x_values[column].std(ddof=0)) > 0
    ]
    if not valid_features:
        raise ValueError("No valid numeric predictors remain after cleaning.")

    x_values = x_values[valid_features]
    y_values = numeric[target.column].astype(float)
    split_index = max(1, min(len(x_values) - 1, int(len(x_values) * train_fraction)))

    x_train = x_values.iloc[:split_index]
    x_test = x_values.iloc[split_index:]
    y_train = y_values.iloc[:split_index].to_numpy(dtype=float)
    y_test = y_values.iloc[split_index:].to_numpy(dtype=float)

    means = x_train.mean()
    stds = x_train.std(ddof=0).replace(0, 1)
    x_train_scaled = (x_train - means) / stds
    x_test_scaled = (x_test - means) / stds

    design = np.column_stack([np.ones(len(x_train_scaled)), x_train_scaled.to_numpy(dtype=float)])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    weights = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y_train

    train_pred = design @ weights
    test_design = np.column_stack([np.ones(len(x_test_scaled)), x_test_scaled.to_numpy(dtype=float)])
    test_pred = test_design @ weights
    metrics = _regression_metrics(y_train, train_pred, y_test, test_pred)
    metrics["filas_entrenamiento"] = float(len(x_train))
    metrics["filas_validacion"] = float(len(x_test))
    metrics["variables"] = float(len(valid_features))

    return LivePredictionModel(
        target=target,
        features=valid_features,
        coefficients=weights[1:],
        intercept=float(weights[0]),
        means=means,
        stds=stds,
        defaults=defaults[valid_features],
        metrics=metrics,
    )


def predict_live(model: LivePredictionModel, values: dict[str, float]) -> float:
    """Predict one scenario with a fitted live baseline.

    Args:
        model: Fitted live prediction model.
        values: Mapping from feature name to user-provided value.

    Returns:
        Numeric prediction in the target's original scale.
    """

    row = pd.Series({feature: values.get(feature, model.defaults[feature]) for feature in model.features})
    row = row.astype(float).fillna(model.defaults)
    scaled = (row - model.means[model.features]) / model.stds[model.features]
    prediction = float(model.intercept + np.dot(scaled.to_numpy(dtype=float), model.coefficients))
    return prediction


def feature_influence(model: LivePredictionModel, top_n: int = 10) -> pd.DataFrame:
    """Return the strongest standardized coefficients for explanation.

    Args:
        model: Fitted live prediction model.
        top_n: Number of predictors to return.

    Returns:
        DataFrame sorted by absolute standardized coefficient.
    """

    influence = pd.DataFrame(
        {
            "Variable": model.features,
            "Capa": [feature_group(feature) for feature in model.features],
            "Coeficiente": model.coefficients,
        }
    )
    influence["Impacto_abs"] = influence["Coeficiente"].abs()
    return influence.sort_values("Impacto_abs", ascending=False).head(top_n)


def irca_risk_label(value: float) -> str:
    """Classify an IRCA value according to common Colombian risk intervals.

    Args:
        value: Predicted or observed IRCA value.

    Returns:
        Human-readable risk level.
    """

    if value <= 5:
        return "Sin riesgo"
    if value <= 14:
        return "Riesgo bajo"
    if value <= 35:
        return "Riesgo medio"
    if value <= 80:
        return "Riesgo alto"
    return "Inviable sanitariamente"


def _regression_metrics(
    y_train: np.ndarray,
    train_pred: np.ndarray,
    y_test: np.ndarray,
    test_pred: np.ndarray,
) -> dict[str, float]:
    """Compute train and validation metrics for regression baselines.

    Args:
        y_train: Observed training target values.
        train_pred: Training predictions.
        y_test: Observed validation target values.
        test_pred: Validation predictions.

    Returns:
        Dictionary with MAE, RMSE and R2 metrics.
    """

    return {
        "mae_train": _mae(y_train, train_pred),
        "rmse_train": _rmse(y_train, train_pred),
        "r2_train": _r2(y_train, train_pred),
        "mae_validacion": _mae(y_test, test_pred),
        "rmse_validacion": _rmse(y_test, test_pred),
        "r2_validacion": _r2(y_test, test_pred),
    }


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error."""

    return float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else float("nan")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute root mean squared error."""

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) else float("nan")


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute coefficient of determination."""

    if len(y_true) < 2:
        return float("nan")
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denominator == 0:
        return float("nan")
    numerator = float(np.sum((y_true - y_pred) ** 2))
    return 1 - numerator / denominator
