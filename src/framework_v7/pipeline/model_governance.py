"""Model-governance helpers extracted from notebook C17.

Notebook C17 is mainly methodological memory. This module turns its governance
content into reusable tables that can be rendered by Streamlit, scripts or
future notebooks without duplicating markdown cells.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from framework_v7.paths import NOTEBOOKS_DIR


MODEL_GOVERNANCE_NOTEBOOK = NOTEBOOKS_DIR / "C17_GOBIERNO_MODELOS" / "FW7_C17_Gobierno_modelos.ipynb"


def governed_models() -> pd.DataFrame:
    """Return the current model-governance catalog.

    Returns:
        pd.DataFrame: One row per governed model with target, problem type and
        intended purpose.
    """

    return pd.DataFrame(
        [
            {
                "Modelo": "Exp01-V3",
                "Variable_objetivo": "irca",
                "Tipo_problema": "Clasificacion",
                "Proposito": "Prediccion del riesgo sanitario asociado a la calidad del agua.",
            },
            {
                "Modelo": "Exp04",
                "Variable_objetivo": "VolumenUtilDiarioMasa",
                "Tipo_problema": "Regresion",
                "Proposito": "Prediccion del volumen util disponible del sistema hidrico.",
            },
        ]
    )


def reference_model_identity() -> pd.DataFrame:
    """Return the identity table for the C17 reference model.

    Returns:
        pd.DataFrame: Key-value metadata for experiment ``Exp01-V3``.
    """

    rows = [
        ("Experimento", "Exp01-V3"),
        ("Dominio", "Gestion hidrica"),
        ("Variable objetivo", "irca"),
        ("Tipo de problema", "Clasificacion"),
        ("Modelo", "LSTM"),
        ("Ventana temporal", "12"),
        ("Horizonte", "1"),
        ("Variables predictoras", "8"),
        ("Metodo de transformacion", "Escalado MinMax"),
        ("Numero de secuencias", "516"),
        ("Tensor de entrada", "tensor_X.npy"),
        ("Tensor objetivo", "tensor_y.npy"),
        ("Modelo fisico", "modelo_Exp01-V3_irca.keras"),
        ("Notebook de gobierno", "FW7_C17_Gobierno_modelos.ipynb"),
    ]
    return pd.DataFrame(rows, columns=["Elemento", "Identificacion"])


def model_artifact_traceability() -> pd.DataFrame:
    """Return the main artifact traceability matrix for C17.

    Returns:
        pd.DataFrame: Governance artifact catalog with category, artifact and
        lifecycle function.
    """

    rows = [
        ("Datos", "dataset_machine_learning.parquet", "Dataset de entrada al proceso de Machine Learning."),
        (
            "Transformacion",
            "dataset_machine_learning_transformado.parquet",
            "Datos despues de la transformacion aplicada.",
        ),
        ("Transformacion", "scaler.pkl", "Objeto usado para conservar la transformacion de variables."),
        ("Secuencias", "secuencias_X.npy", "Secuencias temporales de entrada del modelo."),
        ("Secuencias", "secuencias_y.npy", "Variable objetivo asociada a las secuencias."),
        ("Modelado", "tensor_X.npy", "Tensor de entrada usado durante modelado."),
        ("Modelado", "tensor_y.npy", "Tensor correspondiente a la variable objetivo."),
        ("Metadata", "metadata_tensor.csv", "Descripcion de secuencias y tensores."),
        ("Modelo", "modelo_Exp01-V3_irca.keras", "Artefacto fisico del modelo entrenado."),
        ("Evaluacion", "predicciones.csv", "Resultados de prediccion generados por el modelo."),
        ("Diagnostico", "diagnostico_modelo_Exp01-V3.csv", "Evaluacion tecnica del desempeno."),
        ("Interpretacion", "resumen_experimento.csv", "Lectura tecnica y sistemica del experimento."),
    ]
    return pd.DataFrame(rows, columns=["Categoria", "Artefacto", "Funcion"])


def governance_lifecycle() -> pd.DataFrame:
    """Return lifecycle stages governed by C17.

    Returns:
        pd.DataFrame: Stage-level traceability matrix from data to app usage.
    """

    rows = [
        ("Datos de origen", "C09", "Consolidacion del Dataset Maestro", "Dataset Maestro"),
        ("Preparacion ML", "C12", "Seleccion de variables para Machine Learning", "Dataset ML"),
        ("Diseno experimental", "C13", "Definicion de experimentos y configuracion", "Catalogo experimental"),
        ("Transformacion", "C13", "Construccion de secuencias temporales", "Scaler, secuencias y metadata"),
        ("Modelado", "C14", "Construccion y entrenamiento de modelos", "Tensores y modelos keras"),
        ("Evaluacion", "C15", "Medicion del desempeno", "Metricas y predicciones"),
        ("Interpretacion", "C16", "Lectura tecnica de resultados", "Resumenes interpretativos"),
        ("Gobierno", "C17", "Versionamiento, trazabilidad y control", "Notebook C17"),
        ("Aplicacion", "Streamlit", "Visualizacion e inferencia exploratoria", "App y simulador live"),
    ]
    return pd.DataFrame(rows, columns=["Etapa", "Componente", "Funcion", "Evidencia"])


def model_architecture_summary() -> pd.DataFrame:
    """Return the reference LSTM architecture documented in C17.

    Returns:
        pd.DataFrame: Layer-level architecture summary.
    """

    rows = [
        (1, "LSTM", "(None, 32)", 5248),
        (2, "Dropout", "(None, 32)", 0),
        (3, "Dense", "(None, 1)", 33),
    ]
    return pd.DataFrame(rows, columns=["Capa", "Tipo", "Salida", "Parametros"])


def governance_status() -> pd.DataFrame:
    """Return the current governance-status checklist.

    Returns:
        pd.DataFrame: Status of governance evidence and pending controls.
    """

    rows = [
        ("Evaluacion original de Exp01-V3", "Realizada"),
        ("Evaluacion original de Exp04", "Realizada"),
        ("Metricas de desempeno", "Registradas"),
        ("Diagnostico de modelos", "Registrado"),
        ("Recomendaciones", "Registradas"),
        ("Control de fuga de informacion", "Considerado dentro del proceso"),
        ("Validacion temporal adicional", "Pendiente de fortalecimiento"),
        ("Variabilidad entre particiones temporales", "Pendiente de fortalecimiento"),
        ("Intervalos de confianza", "Pendiente de fortalecimiento"),
        ("Integracion de evidencia adicional al Gobierno", "Pendiente"),
    ]
    return pd.DataFrame(rows, columns=["Elemento", "Estado"])


def governance_risk_controls() -> pd.DataFrame:
    """Return governance risks and the associated control actions.

    Returns:
        pd.DataFrame: Risk-control matrix for safe model use.
    """

    rows = [
        (
            "Fuga de informacion",
            "Uso directo o indirecto de informacion no disponible al momento de la prediccion.",
            "Revision de variables, transformacion y separacion temporal.",
        ),
        (
            "Sobreajuste",
            "El modelo aprende patrones especificos que no generalizan.",
            "Evaluacion sobre datos no usados y analisis de curvas de entrenamiento.",
        ),
        (
            "Cambio de distribucion",
            "Las caracteristicas de los datos pueden cambiar con el tiempo.",
            "Monitoreo periodico y evaluacion de drift.",
        ),
        (
            "Datos fuera del dominio",
            "Ingreso de valores fuera de las condiciones observadas durante entrenamiento.",
            "Validacion de rangos y advertencias en inferencia.",
        ),
        (
            "Datos faltantes",
            "Ausencia de variables necesarias para generar una prediccion.",
            "Validacion de entradas y manejo de excepciones.",
        ),
        (
            "Cambio metodologico",
            "Modificacion de variables, transformaciones o arquitectura sin control.",
            "Versionamiento de experimentos y modelos.",
        ),
        (
            "Interpretacion incorrecta",
            "Uso de una prediccion como certeza absoluta.",
            "Presentacion de resultados junto con limitaciones.",
        ),
        (
            "Deriva del modelo",
            "Perdida progresiva de desempeno despues del despliegue.",
            "Revision periodica y criterios de reentrenamiento.",
        ),
        (
            "Uso no previsto",
            "Uso del modelo en decisiones distintas al proposito definido.",
            "Definicion explicita del proposito y restricciones.",
        ),
    ]
    return pd.DataFrame(rows, columns=["Riesgo", "Descripcion", "Control"])


def governance_risk_matrix() -> pd.DataFrame:
    """Return risk levels documented by C17.

    Returns:
        pd.DataFrame: Probability, impact and control level per risk.
    """

    rows = [
        ("Fuga de informacion", "Media", "Alto", "Alto"),
        ("Sobreajuste", "Media", "Alto", "Alto"),
        ("Cambio de distribucion", "Media", "Alto", "Alto"),
        ("Datos fuera del dominio", "Media", "Alto", "Alto"),
        ("Datos faltantes", "Media", "Medio", "Medio"),
        ("Drift del modelo", "Media", "Alto", "Alto"),
        ("Uso no previsto", "Media", "Alto", "Alto"),
        ("Interpretacion incorrecta", "Media", "Alto", "Alto"),
        ("Modificacion sin versionamiento", "Baja", "Alto", "Alto"),
        ("Fallos de inferencia", "Baja/Media", "Medio", "Medio"),
    ]
    return pd.DataFrame(rows, columns=["Riesgo", "Probabilidad", "Impacto", "Nivel_control"])


def notebook_markdown_sections(notebook_path: Path | None = None) -> pd.DataFrame:
    """Extract markdown sections from the C17 notebook.

    Args:
        notebook_path (Path | None): Optional notebook path. When omitted, the
            canonical C17 notebook path is used.

    Returns:
        pd.DataFrame: Markdown section title and content. Returns an empty
        DataFrame when the notebook is unavailable.
    """

    path = notebook_path or MODEL_GOVERNANCE_NOTEBOOK
    if not path.exists():
        return pd.DataFrame(columns=["Seccion", "Contenido"])

    notebook = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        content = "".join(cell.get("source", [])).strip()
        if not content:
            continue
        title = content.splitlines()[0].replace("#", "").replace("*", "").strip()
        rows.append({"Seccion": title, "Contenido": content})
    return pd.DataFrame(rows)


def governance_summary() -> pd.DataFrame:
    """Build a compact C17 summary for pipeline checks.

    Returns:
        pd.DataFrame: Summary indicators for model governance.
    """

    return pd.DataFrame(
        [
            {"Indicador": "modelos_gobernados", "Valor": len(governed_models())},
            {"Indicador": "artefactos_trazados", "Valor": len(model_artifact_traceability())},
            {"Indicador": "etapas_ciclo_vida", "Valor": len(governance_lifecycle())},
            {"Indicador": "riesgos_controlados", "Valor": len(governance_risk_controls())},
            {"Indicador": "secciones_notebook", "Valor": len(notebook_markdown_sections())},
        ]
    )
