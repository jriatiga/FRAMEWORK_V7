"""Streamlit view functions for FRAMEWORK V7.

Each function renders one high-level screen in the application. The view layer
depends on data-access, profiling and visualization helpers, but it does not
define business catalog constants or read files directly except through helper
functions.
"""

from __future__ import annotations

from urllib.parse import quote

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .catalog import EXPERIMENT_DESIGN_FILES, FEATURE_GROUPS, LAYER_CATALOG, MASTER_FILES, SUPPORT_FILES
from .data_access import ProjectData, load_csv, load_excel, read_text
from .layers import LAYER_MODULES
from .paths import (
    BASE_DIR,
    COVERAGE_PATH,
    EVALUATIONS_DIR,
    EXPERIMENT_DESIGN_DIR,
    INTERPRETATION_DIR,
    MACHINE_LEARNING_DIR,
    MASTER_PATH,
    ML_DATASET_PATH,
    MODEL_ACCURACY_IMAGE_PATH,
    MODEL_CARDS_DIR,
    MODEL_DIAGNOSTIC_PATH,
    MODEL_LOSS_IMAGE_PATH,
    MODELING_DIR,
    NOTEBOOKS_DIR,
    PREDICTIONS_PATH,
    rel,
)
from .pipeline.evaluation import (
    evaluation_artifact_inventory,
    load_prediction_metadata,
    load_predictions,
    prediction_distribution,
    summarize_evaluation_experiments,
)
from .pipeline.interpretation import (
    dimension_coverage,
    interpretation_artifact_inventory,
    load_interpretation_summary,
    summarize_interpretation_experiments,
)
from .pipeline.live_prediction import (
    IRCA_TARGET_LABEL,
    available_live_targets,
    feature_influence,
    feature_profile,
    fit_live_ridge_model,
    irca_risk_label,
    predict_live,
)
from .pipeline.machine_learning import (
    load_sequence_metadata,
    load_transformed_dataset,
    ml_artifact_inventory,
    summarize_ml_experiments,
)
from .pipeline.modeling import (
    load_model_diagnostic,
    load_model_record,
    load_model_recommendations,
    modeling_artifact_inventory,
    summarize_modeling_experiments,
)
from .profiling import find_date_column, layer_summary, normalize_01, quality_badge
from .utils import format_metric_date
from .visualizations import (
    render_dataset_metrics,
    render_layer_images,
    render_missing_file,
    render_missing_profile,
    render_numeric_overview,
    render_system_map,
    render_time_series,
)

GITHUB_RAW_BASE_URL = "https://github.com/josedanielbg/FRAMEWORK_V7/raw/main"


def _experiment_names(*frames: pd.DataFrame) -> list[str]:
    """Return sorted experiment identifiers from summary tables.

    Args:
        *frames: DataFrames that may contain an ``Experimento`` column.

    Returns:
        Sorted list of unique experiment names.
    """

    names = set()
    for frame in frames:
        if not frame.empty and "Experimento" in frame.columns:
            names.update(frame["Experimento"].dropna().astype(str))
    return sorted(names)


def _numeric_metric_frame(summary: pd.DataFrame) -> pd.DataFrame:
    """Convert wide experiment metrics into a long chart-ready table.

    Args:
        summary: Experiment summary with metric columns.

    Returns:
        DataFrame with ``Experimento``, ``Metrica`` and ``Valor`` columns.
    """

    metric_columns = ["Accuracy", "Precision", "Recall", "F1", "MAE", "RMSE", "MAPE", "R2"]
    available = [column for column in metric_columns if column in summary.columns]
    if summary.empty or not available or "Experimento" not in summary.columns:
        return pd.DataFrame(columns=["Experimento", "Metrica", "Valor"])

    metrics = summary[["Experimento", *available]].melt(
        id_vars="Experimento",
        var_name="Metrica",
        value_name="Valor",
    )
    metrics["Valor"] = pd.to_numeric(metrics["Valor"], errors="coerce")
    return metrics.dropna(subset=["Valor"])


def _prediction_view(experiment: str) -> pd.DataFrame:
    """Load prediction rows and attach the experiment name.

    Args:
        experiment: Experiment identifier.

    Returns:
        Prediction DataFrame prepared for plotting.
    """

    predictions = load_predictions(experiment).copy()
    if predictions.empty or "Prediccion" not in predictions.columns:
        return pd.DataFrame()
    if "Registro" not in predictions.columns:
        predictions["Registro"] = range(1, len(predictions) + 1)
    predictions["Prediccion"] = pd.to_numeric(predictions["Prediccion"], errors="coerce")
    predictions["Experimento"] = experiment
    predictions["Tendencia"] = predictions["Prediccion"].rolling(12, min_periods=1).mean()
    predictions["Intensidad"] = normalize_01(predictions["Prediccion"])
    return predictions


def _artifact_inventory() -> pd.DataFrame:
    """Build a consolidated artifact inventory for executed experiments.

    Returns:
        DataFrame with stage, experiment, file path, format and size.
    """

    inventories = [
        ("C13 Machine Learning", ml_artifact_inventory()),
        ("C14 Modelado", modeling_artifact_inventory()),
        ("C15 Evaluacion", evaluation_artifact_inventory()),
        ("C16 Interpretacion", interpretation_artifact_inventory()),
    ]
    frames = []
    for stage, inventory in inventories:
        if inventory.empty:
            continue
        frame = inventory.copy()
        frame.insert(0, "Etapa", stage)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _model_card_inventory() -> pd.DataFrame:
    """Build an inventory of available model-card PDF files.

    Returns:
        DataFrame with experiment label, file name, repository path and size.
    """

    if not MODEL_CARDS_DIR.exists():
        return pd.DataFrame(columns=["Experimento", "Archivo", "Ruta", "Tamano_MB"])

    rows = []
    for pdf_path in sorted(MODEL_CARDS_DIR.glob("*.pdf")):
        experiment = pdf_path.stem.replace("_", "-")
        rows.append(
            {
                "Experimento": experiment,
                "Archivo": pdf_path.name,
                "Ruta": rel(pdf_path),
                "Tamano_MB": round(pdf_path.stat().st_size / (1024 * 1024), 2),
            }
        )
    return pd.DataFrame(rows)


def _github_raw_url(file_path) -> str:
    """Build a direct GitHub URL for a repository file.

    Args:
        file_path: Local path inside the repository.

    Returns:
        Direct URL that opens or downloads the file from GitHub.
    """

    relative_path = rel(file_path).replace("\\", "/")
    return f"{GITHUB_RAW_BASE_URL}/{quote(relative_path, safe='/')}"


def render_sidebar(data: ProjectData) -> str:
    """Render the sidebar and return the selected section.

    Args:
        data: Loaded project datasets and metadata.

    Returns:
        Name of the selected dashboard section.
    """

    with st.sidebar:
        st.title("FRAMEWORK V7")
        section = st.radio(
            "Vista",
            [
                "Dashboard",
                "Experimentos",
                "Prediccion live",
                "Diseno experimental",
                "Datasets por capas",
                "Dataset maestro",
                "Notebooks",
            ],
        )
        st.divider()
        evaluation_summary = summarize_evaluation_experiments()
        st.caption("Experimentos evaluados")
        st.write(str(len(evaluation_summary)) if not evaluation_summary.empty else "1")
        st.caption("Base historica")
        st.write(data.meta.get("Experimento", "Exp01"))
        st.caption("Fecha ejecucion")
        st.write(data.meta.get("Fecha Ejecucion", "-"))
    return section


def render_dashboard(data: ProjectData) -> None:
    """Render the executive multicapa dashboard.

    Args:
        data: Loaded project datasets and metadata.

    Returns:
        None.
    """

    st.subheader("Resumen ejecutivo")
    layers = layer_summary()
    evaluation_summary = summarize_evaluation_experiments()
    interpretation_summary = summarize_interpretation_experiments()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Capas sistemicas", f"{int(layers['Disponible'].sum())}/{len(layers)}")
    c2.metric("Dataset maestro", f"{len(data.master):,} filas")
    c3.metric("Variables maestro", f"{data.master.shape[1]:,}")
    c4.metric("Experimentos evaluados", f"{len(evaluation_summary):,}")

    tab_map, tab_pred, tab_layers, tab_quality = st.tabs(
        ["Mapa del sistema", "Predicciones", "Capas", "Calidad de datos"]
    )
    with tab_map:
        render_system_map()
        st.dataframe(
            layers[["Capa", "Rol sistemico", "Filas", "Columnas", "Nulos"]],
            use_container_width=True,
            hide_index=True,
        )

    with tab_pred:
        experiments = _experiment_names(evaluation_summary)
        prediction_frames = [_prediction_view(experiment) for experiment in experiments]
        prediction_frames = [frame for frame in prediction_frames if not frame.empty]
        if not prediction_frames and data.predictions.empty:
            render_missing_file(PREDICTIONS_PATH)
        else:
            view = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else data.predictions.copy()
            fig = px.line(
                view,
                x="Registro",
                y="Prediccion",
                color="Experimento" if "Experimento" in view.columns else None,
                title="Predicciones por experimento",
            )
            fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)
            if not interpretation_summary.empty:
                metrics = _numeric_metric_frame(interpretation_summary)
                if not metrics.empty:
                    fig = px.bar(
                        metrics,
                        x="Experimento",
                        y="Valor",
                        color="Metrica",
                        barmode="group",
                        title="Metricas consolidadas de interpretacion",
                    )
                    fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
                    st.plotly_chart(fig, use_container_width=True)

    with tab_layers:
        fig = px.bar(
            layers,
            x="Capa",
            y="Filas",
            color="Nulos",
            title="Volumen de datos por capa",
            color_continuous_scale=["#2A9D8F", "#F4A261", "#E76F51"],
        )
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab_quality:
        if not data.coverage.empty:
            st.dataframe(data.coverage, use_container_width=True, hide_index=True)
        else:
            render_missing_profile(data.master)


def render_experiments(data: ProjectData) -> None:
    """Render a multi-experiment analysis center.

    Args:
        data: Loaded project datasets and metadata.

    Returns:
        None.
    """

    st.subheader("Centro de experimentos")
    ml_summary = summarize_ml_experiments()
    modeling_summary = summarize_modeling_experiments()
    evaluation_summary = summarize_evaluation_experiments()
    interpretation_summary = summarize_interpretation_experiments()
    experiments = _experiment_names(ml_summary, modeling_summary, evaluation_summary, interpretation_summary)

    if not experiments:
        render_experiment(data)
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Experimentos", f"{len(experiments):,}")
    c2.metric("Con predicciones", f"{len(evaluation_summary):,}")
    c3.metric("Con modelado", f"{len(modeling_summary):,}")
    c4.metric("Interpretados", f"{len(interpretation_summary):,}")

    selected_experiment = st.selectbox("Experimento", experiments, key="experiment_center_selected")
    (
        tab_compare,
        tab_detail,
        tab_predictions,
        tab_modeling,
        tab_model_cards,
        tab_interpretation,
        tab_artifacts,
    ) = st.tabs(
        [
            "Comparativo",
            "Detalle",
            "Predicciones",
            "Modelado",
            "Model cards",
            "Interpretacion",
            "Artefactos",
        ]
    )

    with tab_compare:
        left, right = st.columns([1, 1])
        with left:
            if evaluation_summary.empty:
                render_missing_file(EVALUATIONS_DIR)
            else:
                fig = px.bar(
                    evaluation_summary,
                    x="Experimento",
                    y="Predicciones",
                    color="Variable_Objetivo",
                    title="Predicciones generadas por experimento",
                )
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(fig, use_container_width=True)
        with right:
            metrics = _numeric_metric_frame(interpretation_summary)
            if metrics.empty:
                render_missing_file(INTERPRETATION_DIR)
            else:
                fig = px.bar(
                    metrics,
                    x="Experimento",
                    y="Valor",
                    color="Metrica",
                    barmode="group",
                    title="Metricas comparables C16",
                )
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(fig, use_container_width=True)

        summary_tabs = st.tabs(["Preparacion ML", "Modelado", "Evaluacion", "Interpretacion"])
        with summary_tabs[0]:
            st.caption(rel(MACHINE_LEARNING_DIR / "Transformaciones"))
            st.dataframe(ml_summary, use_container_width=True, hide_index=True)
        with summary_tabs[1]:
            st.caption(rel(MODELING_DIR))
            st.dataframe(modeling_summary, use_container_width=True, hide_index=True)
        with summary_tabs[2]:
            st.caption(rel(EVALUATIONS_DIR))
            st.dataframe(evaluation_summary, use_container_width=True, hide_index=True)
        with summary_tabs[3]:
            st.caption(rel(INTERPRETATION_DIR))
            st.dataframe(interpretation_summary, use_container_width=True, hide_index=True)

    with tab_detail:
        metadata = load_prediction_metadata(selected_experiment)
        sequence_metadata = load_sequence_metadata(selected_experiment)
        ml_dataset = load_transformed_dataset(selected_experiment)
        model_record = load_model_record(selected_experiment)
        predictions = load_predictions(selected_experiment)
        target = metadata.get("Variable Objetivo", sequence_metadata.get("Variable Objetivo", "-"))

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Variable objetivo", target)
        d2.metric("Ventana", metadata.get("Ventana", sequence_metadata.get("Ventana", "-")))
        d3.metric("Filas ML", f"{len(ml_dataset):,}")
        d4.metric("Predicciones", f"{len(predictions):,}")

        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Metadata de prediccion**")
            metadata_table = pd.DataFrame(
                [{"Parametro": key, "Valor": value} for key, value in metadata.items()]
            )
            st.dataframe(metadata_table, use_container_width=True, hide_index=True)
        with right:
            st.markdown("**Registro de modelado**")
            if model_record.empty:
                render_missing_file(MODELING_DIR / "Modelos" / selected_experiment)
            else:
                st.dataframe(model_record, use_container_width=True, hide_index=True)

        if not ml_dataset.empty:
            with st.expander("Dataset transformado C13", expanded=False):
                render_dataset_metrics(ml_dataset)
                st.dataframe(ml_dataset.head(300), use_container_width=True, hide_index=True)

    with tab_predictions:
        prediction_view = _prediction_view(selected_experiment)
        if prediction_view.empty:
            render_missing_file(EVALUATIONS_DIR / selected_experiment / "predicciones.csv")
        else:
            p1, p2, p3, p4 = st.columns(4)
            distribution = prediction_distribution(prediction_view)
            row = distribution.iloc[0].to_dict() if not distribution.empty else {}
            p1.metric("Conteo", f"{int(row.get('Conteo', len(prediction_view))):,}")
            p2.metric("Media", f"{row.get('Media', 0):.3f}")
            p3.metric("Minimo", f"{row.get('Minimo', 0):.3f}")
            p4.metric("Maximo", f"{row.get('Maximo', 0):.3f}")

            left, right = st.columns([2, 1])
            with left:
                fig = px.line(
                    prediction_view,
                    x="Registro",
                    y=["Prediccion", "Tendencia"],
                    title=f"Serie de prediccion {selected_experiment}",
                )
                fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(fig, use_container_width=True)
            with right:
                prediction_view["Categoria"] = prediction_view["Intensidad"].apply(quality_badge)
                counts = prediction_view["Categoria"].value_counts().reset_index()
                counts.columns = ["Categoria", "Registros"]
                fig = px.pie(
                    counts,
                    names="Categoria",
                    values="Registros",
                    hole=0.45,
                    title="Intensidad relativa",
                )
                fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(prediction_view, use_container_width=True, hide_index=True)

    with tab_modeling:
        diagnostic = load_model_diagnostic(selected_experiment)
        recommendations = load_model_recommendations(selected_experiment)
        if diagnostic.empty:
            render_missing_file(MODELING_DIR / "Diagnosticos" / selected_experiment)
        else:
            metric_values = diagnostic.copy()
            metric_values["Valor_Numerico"] = pd.to_numeric(metric_values["Valor"], errors="coerce")
            chart_values = metric_values.dropna(subset=["Valor_Numerico"])
            if not chart_values.empty:
                fig = px.bar(
                    chart_values,
                    x="Indicador",
                    y="Valor_Numerico",
                    color="Estado" if "Estado" in chart_values.columns else None,
                    title=f"Diagnostico del modelo {selected_experiment}",
                    color_discrete_map={
                        "Excelente": "#2A9D8F",
                        "Aceptable": "#E9C46A",
                        "Baja": "#F4A261",
                        "Critico": "#E76F51",
                        "Crítico": "#E76F51",
                    },
                )
                fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(diagnostic, use_container_width=True, hide_index=True)

        if not recommendations.empty:
            st.markdown("**Recomendaciones**")
            st.dataframe(recommendations, use_container_width=True, hide_index=True)

        metric_images = sorted((MODELING_DIR / "Metricas" / selected_experiment).glob("*.png"))
        if metric_images:
            cols = st.columns(min(3, len(metric_images)))
            for index, image in enumerate(metric_images):
                with cols[index % len(cols)]:
                    st.image(str(image), caption=image.name, use_container_width=True)

    with tab_model_cards:
        cards = _model_card_inventory()
        if cards.empty:
            render_missing_file(MODEL_CARDS_DIR)
        else:
            normalized_experiment = selected_experiment.replace("_", "-")
            options = cards["Archivo"].tolist()
            matched = cards[cards["Experimento"].astype(str) == normalized_experiment]
            default_index = 0
            if not matched.empty:
                default_file = matched["Archivo"].iloc[0]
                default_index = options.index(default_file)

            selected_card = st.selectbox(
                "Model card",
                options,
                index=default_index,
                key=f"model_card_{selected_experiment}",
            )
            card_path = MODEL_CARDS_DIR / selected_card
            c1, c2, c3 = st.columns(3)
            card_row = cards[cards["Archivo"] == selected_card].iloc[0]
            c1.metric("Experimento", card_row["Experimento"])
            c2.metric("Archivo", selected_card)
            c3.metric("Tamano", f"{float(card_row['Tamano_MB']):.2f} MB")
            st.caption(rel(card_path))

            if not card_path.exists():
                render_missing_file(card_path)
            else:
                st.info(
                    "La model card se abre fuera del visor embebido para evitar "
                    "bloqueos del navegador en Streamlit Cloud."
                )
                action_col1, action_col2 = st.columns(2)
                with action_col1:
                    st.download_button(
                        "Descargar model card",
                        data=card_path.read_bytes(),
                        file_name=selected_card,
                        mime="application/pdf",
                        key=f"download_model_card_{selected_card}",
                    )
                with action_col2:
                    st.link_button("Abrir PDF en GitHub", _github_raw_url(card_path))

            st.dataframe(cards, use_container_width=True, hide_index=True)

    with tab_interpretation:
        summary = load_interpretation_summary(selected_experiment)
        interpreted = interpretation_summary[
            interpretation_summary.get("Experimento", pd.Series(dtype=str)).astype(str) == selected_experiment
        ]
        if summary.empty and interpreted.empty:
            render_missing_file(INTERPRETATION_DIR / selected_experiment / "resumen_experimento.csv")
        else:
            view = interpreted if not interpreted.empty else summary
            st.dataframe(view, use_container_width=True, hide_index=True)
            if "Interpretacion_Tecnica" in view.columns:
                st.info(str(view["Interpretacion_Tecnica"].iloc[0]))

            sequence_metadata = load_sequence_metadata(selected_experiment)
            variables = str(sequence_metadata.get("Variables Predictoras", "")).split(";")
            variables = [variable.strip() for variable in variables if variable.strip()]
            coverage = dimension_coverage(variables)
            if not coverage.empty:
                fig = px.bar(
                    coverage,
                    x="Dimension",
                    y="Variables",
                    color="Dimension",
                    title="Cobertura sistemica de variables predictoras",
                )
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(coverage, use_container_width=True, hide_index=True)

    with tab_artifacts:
        inventory = _artifact_inventory()
        if inventory.empty:
            render_missing_file(BASE_DIR / "DATA")
        else:
            filtered = inventory[inventory["Experimento"].astype(str) == selected_experiment]
            if filtered.empty:
                filtered = inventory
            counts = filtered.groupby(["Etapa", "Formato"]).size().reset_index(name="Archivos")
            fig = px.bar(
                counts,
                x="Etapa",
                y="Archivos",
                color="Formato",
                barmode="group",
                title="Artefactos disponibles por etapa",
            )
            fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(filtered, use_container_width=True, hide_index=True)


def render_experiment(data: ProjectData) -> None:
    """Render the Exp01 experiment view.

    Args:
        data: Loaded project datasets and metadata.

    Returns:
        None.
    """

    tab_summary, tab_series, tab_dist, tab_ml, tab_metadata = st.tabs(
        ["Resumen", "Serie", "Distribucion", "Variables ML", "Metadata"]
    )
    with tab_summary:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Objetivo", data.meta.get("Variable Objetivo", "irca"))
        col2.metric("Modelo", data.meta.get("Modelo", "-"))
        col3.metric("Ventana", data.meta.get("Ventana", "12"))
        col4.metric("Predictoras", data.meta.get("Variables Predictoras", "-"))
        if not data.predictions.empty and "Prediccion" in data.predictions.columns:
            values = pd.to_numeric(data.predictions["Prediccion"], errors="coerce")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Media", f"{values.mean():.3f}")
            c2.metric("Minimo", f"{values.min():.3f}")
            c3.metric("Maximo", f"{values.max():.3f}")
            c4.metric("Desviacion", f"{values.std():.3f}")

    with tab_series:
        if data.predictions.empty:
            render_missing_file(PREDICTIONS_PATH)
        else:
            view = data.predictions.copy()
            view["Prediccion"] = pd.to_numeric(view["Prediccion"], errors="coerce")
            view["Normalizada"] = normalize_01(view["Prediccion"])
            view["Categoria"] = view["Normalizada"].apply(quality_badge)
            selected = st.slider(
                "Rango de registros",
                int(view["Registro"].min()),
                int(view["Registro"].max()),
                (int(view["Registro"].min()), int(view["Registro"].max())),
            )
            view = view[(view["Registro"] >= selected[0]) & (view["Registro"] <= selected[1])]
            fig = px.area(
                view,
                x="Registro",
                y="Normalizada",
                color="Categoria",
                title="Intensidad relativa de prediccion",
            )
            fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(view, use_container_width=True, hide_index=True)

    with tab_dist:
        if not data.predictions.empty and "Prediccion" in data.predictions.columns:
            values_df = data.predictions.copy()
            values_df["Prediccion"] = pd.to_numeric(values_df["Prediccion"], errors="coerce")
            left, right = st.columns(2)
            with left:
                fig = px.histogram(values_df, x="Prediccion", nbins=35, marginal="box", title="Histograma con caja")
                fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(fig, use_container_width=True)
            with right:
                values_df["Categoria"] = normalize_01(values_df["Prediccion"]).apply(quality_badge)
                counts = values_df["Categoria"].value_counts().reset_index()
                counts.columns = ["Categoria", "Registros"]
                fig = px.pie(counts, names="Categoria", values="Registros", hole=0.45, title="Categorias relativas")
                fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(fig, use_container_width=True)

    with tab_ml:
        if data.ml_dataset.empty:
            render_missing_file(ML_DATASET_PATH)
        else:
            render_dataset_metrics(data.ml_dataset)
            render_numeric_overview(data.ml_dataset, "exp_ml")

    with tab_metadata:
        st.dataframe(data.metadata, use_container_width=True, hide_index=True)
        if not data.diagnostic.empty:
            st.markdown("**Diagnostico estadistico**")
            st.dataframe(data.diagnostic, use_container_width=True, hide_index=True)


def _format_live_value(value: float, unit: str) -> str:
    """Format live predictions for compact UI metrics.

    Args:
        value: Numeric prediction.
        unit: Target unit.

    Returns:
        Human-readable prediction label.
    """

    if unit == "m3":
        return f"{value:,.0f} {unit}"
    return f"{value:,.2f} {unit}"


def _slider_step(minimum: float, maximum: float) -> float:
    """Choose a practical step for Streamlit numeric controls.

    Args:
        minimum: Observed minimum.
        maximum: Observed maximum.

    Returns:
        Positive step size.
    """

    span = abs(maximum - minimum)
    if span == 0:
        return 1.0
    if span >= 1_000_000:
        return max(1.0, round(span / 100))
    if span >= 100:
        return 1.0
    if span >= 10:
        return 0.1
    return 0.01


def render_live_prediction(data: ProjectData) -> None:
    """Render the interactive live prediction simulator.

    Args:
        data: Loaded project datasets and metadata.

    Returns:
        None.
    """

    st.subheader("Prediccion live")
    if data.master.empty:
        render_missing_file(MASTER_PATH)
        return

    available_targets = available_live_targets(data.master)
    if not available_targets:
        st.warning("No hay variables objetivo suficientes para entrenar el simulador live.")
        return

    selected_target = st.selectbox("Variable objetivo", available_targets)
    try:
        model = fit_live_ridge_model(data.master, selected_target)
    except ValueError as error:
        st.error(str(error))
        return

    st.caption(model.target.description)
    metrics = model.metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas entrenamiento", f"{int(metrics['filas_entrenamiento']):,}")
    c2.metric("Filas validacion", f"{int(metrics['filas_validacion']):,}")
    c3.metric("MAE validacion", f"{metrics['mae_validacion']:,.3f}")
    c4.metric("R2 validacion", f"{metrics['r2_validacion']:,.3f}")

    profile = feature_profile(data.master, model.features)
    if profile.empty:
        st.warning("No fue posible perfilar variables predictoras para el simulador.")
        return

    tab_manual, tab_explain, tab_batch = st.tabs(
        ["Simulador manual", "Explicabilidad", "Carga CSV/Excel"]
    )

    with tab_manual:
        st.markdown("**Escenario multicapa**")
        input_values = {}
        for group in profile["Capa"].drop_duplicates().tolist():
            group_profile = profile[profile["Capa"] == group]
            with st.expander(group, expanded=group in ["Clima y variabilidad", "Calidad y percepcion"]):
                columns = st.columns(2)
                for index, row in enumerate(group_profile.itertuples(index=False)):
                    minimum = float(row.Min)
                    maximum = float(row.Max)
                    default = float(row.Mediana)
                    step = _slider_step(minimum, maximum)
                    with columns[index % 2]:
                        if maximum - minimum > 10_000_000:
                            value = st.number_input(
                                row.Variable,
                                min_value=minimum,
                                max_value=maximum,
                                value=default,
                                step=step,
                                key=f"live_input_{selected_target}_{row.Variable}",
                            )
                        else:
                            value = st.slider(
                                row.Variable,
                                min_value=minimum,
                                max_value=maximum,
                                value=default,
                                step=step,
                                key=f"live_input_{selected_target}_{row.Variable}",
                            )
                        input_values[row.Variable] = float(value)

        prediction = predict_live(model, input_values)
        observed_target = pd.to_numeric(data.master[model.target.column], errors="coerce").dropna()
        reference = float(observed_target.median()) if not observed_target.empty else prediction
        delta = prediction - reference

        result_col, chart_col = st.columns([1, 2])
        with result_col:
            st.metric(
                "Prediccion del escenario",
                _format_live_value(prediction, model.target.unit),
                delta=_format_live_value(delta, model.target.unit),
            )
            if selected_target == IRCA_TARGET_LABEL:
                st.metric("Nivel de riesgo estimado", irca_risk_label(prediction))
            else:
                st.caption("Delta calculado contra la mediana historica del objetivo.")

        with chart_col:
            scenarios = pd.DataFrame(
                [
                    {
                        "Escenario": "Mediana historica",
                        "Prediccion": predict_live(model, model.defaults.to_dict()),
                    },
                    {"Escenario": "Escenario editado", "Prediccion": prediction},
                ]
            )
            fig = px.bar(
                scenarios,
                x="Escenario",
                y="Prediccion",
                color="Escenario",
                text_auto=".2s",
                title="Comparacion de escenario live",
            )
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=55, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        if selected_target == IRCA_TARGET_LABEL:
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=max(0.0, prediction),
                    title={"text": "IRCA estimado"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#2563EB"},
                        "steps": [
                            {"range": [0, 5], "color": "#DCFCE7"},
                            {"range": [5, 14], "color": "#FEF9C3"},
                            {"range": [14, 35], "color": "#FED7AA"},
                            {"range": [35, 80], "color": "#FECACA"},
                            {"range": [80, 100], "color": "#E5E7EB"},
                        ],
                    },
                )
            )
            gauge.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(gauge, use_container_width=True)

    with tab_explain:
        st.markdown("**Variables con mayor influencia estandarizada**")
        influence = feature_influence(model, top_n=12)
        if influence.empty:
            st.info("No hay coeficientes disponibles para explicar el baseline.")
        else:
            fig = px.bar(
                influence.sort_values("Coeficiente"),
                x="Coeficiente",
                y="Variable",
                color="Capa",
                orientation="h",
                title="Coeficientes del baseline live",
            )
            fig.update_layout(height=520, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(influence, use_container_width=True, hide_index=True)
            st.caption(
                "Los coeficientes son estandarizados: valores positivos aumentan "
                "la prediccion y valores negativos la reducen, manteniendo el "
                "resto de variables constantes."
            )

    with tab_batch:
        st.markdown("**Prediccion por archivo**")
        uploaded_file = st.file_uploader(
            "Carga un CSV o Excel con columnas compatibles",
            type=["csv", "xlsx"],
            key=f"live_upload_{selected_target}",
        )
        if uploaded_file is None:
            st.dataframe(
                profile[["Variable", "Capa", "Mediana", "Min", "Max"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            try:
                if uploaded_file.name.lower().endswith(".xlsx"):
                    upload_df = pd.read_excel(uploaded_file)
                else:
                    upload_df = pd.read_csv(uploaded_file)
            except Exception as error:  # pragma: no cover - UI guardrail
                st.error(f"No fue posible leer el archivo: {error}")
                return

            scoring_df = upload_df.copy()
            for feature in model.features:
                if feature not in scoring_df.columns:
                    scoring_df[feature] = model.defaults[feature]
            predictions = scoring_df[model.features].apply(
                lambda row: predict_live(model, row.to_dict()),
                axis=1,
            )
            result = upload_df.copy()
            result[f"Prediccion_{model.target.column}"] = predictions
            if selected_target == IRCA_TARGET_LABEL:
                result["Riesgo_estimado"] = predictions.apply(irca_risk_label)
            st.success(f"Predicciones generadas: {len(result):,}")
            st.dataframe(result, use_container_width=True, hide_index=True)
            st.download_button(
                "Descargar predicciones",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name=f"predicciones_live_{model.target.column}.csv",
                mime="text/csv",
                key=f"download_live_predictions_{selected_target}",
            )


def render_experiment_design(data: ProjectData) -> None:
    """Render the experimental design view.

    Args:
        data: Loaded project datasets and metadata.

    Returns:
        None.
    """

    st.subheader("Diseno experimental")
    readme = read_text(EXPERIMENT_DESIGN_DIR / "README.md")
    if readme:
        st.markdown(readme)

    catalog = data.experiment_design.get("Catalogo de experimentos", pd.DataFrame())
    config = data.experiment_design.get("Configuracion", pd.DataFrame())
    predictors = data.experiment_design.get("Variables predictoras", pd.DataFrame())
    status = data.experiment_design.get("Estado de experimentos", pd.DataFrame())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Experimentos", f"{len(catalog):,}")
    c2.metric("Ejecutados", f"{(catalog.get('Estado', pd.Series(dtype=str)) == 'Ejecutado').sum():,}")
    c3.metric("Pendientes", f"{(catalog.get('Estado', pd.Series(dtype=str)) == 'Pendiente').sum():,}")
    c4.metric("Predictoras", f"{len(predictors):,}")

    tab_map, tab_config, tab_diagnostic, tab_files = st.tabs(
        ["Mapa experimental", "Configuracion", "Diagnostico Exp01", "Archivos"]
    )

    with tab_map:
        if catalog.empty:
            render_missing_file(EXPERIMENT_DESIGN_FILES["Catalogo de experimentos"])
        else:
            left, right = st.columns([1, 2])
            with left:
                selected = st.selectbox("Experimento", catalog["Experimento"].tolist())
                experiment = catalog[catalog["Experimento"] == selected].iloc[0]
                st.metric("Objetivo", experiment.get("Variable_Objetivo", "-"))
                st.metric("Tipo", experiment.get("Tipo_Problema", "-"))
                st.metric("Estado", experiment.get("Estado", "-"))
                st.caption(str(experiment.get("Pregunta_Investigacion", "")))
            with right:
                counts = catalog.groupby(["Tipo_Problema", "Estado"]).size().reset_index(name="Experimentos")
                fig = px.bar(
                    counts,
                    x="Tipo_Problema",
                    y="Experimentos",
                    color="Estado",
                    barmode="group",
                    title="Plan experimental por tipo de problema",
                    color_discrete_map={"Ejecutado": "#2A9D8F", "Pendiente": "#E9C46A"},
                )
                fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(catalog, use_container_width=True, hide_index=True)

    with tab_config:
        config_tab, predictors_tab, criteria_tab, status_tab = st.tabs(
            ["Parametros", "Predictoras", "Criterios", "Estado"]
        )
        with config_tab:
            st.dataframe(config, use_container_width=True, hide_index=True)
        with predictors_tab:
            st.dataframe(predictors, use_container_width=True, hide_index=True)
            if not predictors.empty and "Variable" in predictors.columns:
                st.write(", ".join(predictors["Variable"].astype(str).tolist()))
        with criteria_tab:
            for label in ["Criterios clasificacion", "Criterios regresion"]:
                criteria = data.experiment_design.get(label, pd.DataFrame())
                with st.expander(label, expanded=label == "Criterios clasificacion"):
                    st.dataframe(criteria, use_container_width=True, hide_index=True)
        with status_tab:
            st.dataframe(status, use_container_width=True, hide_index=True)

    with tab_diagnostic:
        diagnostic = data.model_diagnostic.copy()
        recommendations = data.model_recommendations.copy()
        if diagnostic.empty:
            render_missing_file(MODEL_DIAGNOSTIC_PATH)
        else:
            st.markdown("**Resultado del modelo Exp01**")
            metric_values = diagnostic.copy()
            metric_values["Valor_Numerico"] = pd.to_numeric(metric_values["Valor"], errors="coerce")
            chart_values = metric_values.dropna(subset=["Valor_Numerico"])
            if not chart_values.empty:
                fig = px.bar(
                    chart_values,
                    x="Indicador",
                    y="Valor_Numerico",
                    color="Estado",
                    title="Metricas de clasificacion Exp01",
                    color_discrete_map={
                        "Excelente": "#2A9D8F",
                        "Aceptable": "#E9C46A",
                        "Baja": "#F4A261",
                        "Critico": "#E76F51",
                    },
                )
                fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(diagnostic, use_container_width=True, hide_index=True)

        if not recommendations.empty and "Recomendacion" in recommendations.columns:
            st.markdown("**Recomendaciones metodologicas**")
            for recommendation in recommendations["Recomendacion"].dropna().astype(str):
                st.write(recommendation)

        images = [path for path in [MODEL_ACCURACY_IMAGE_PATH, MODEL_LOSS_IMAGE_PATH] if path.exists()]
        if images:
            cols = st.columns(len(images))
            for column, image in zip(cols, images):
                with column:
                    st.image(str(image), caption=image.name, use_container_width=True)

    with tab_files:
        selected_label = st.selectbox("Dataset de diseno", list(EXPERIMENT_DESIGN_FILES))
        selected_path = EXPERIMENT_DESIGN_FILES[selected_label]
        dataset = data.experiment_design.get(selected_label, pd.DataFrame())
        st.caption(rel(selected_path))
        if dataset.empty:
            render_missing_file(selected_path)
        else:
            render_dataset_metrics(dataset)
            st.dataframe(dataset, use_container_width=True, hide_index=True)
            st.download_button(
                "Descargar dataset",
                data=dataset.to_csv(index=False).encode("utf-8"),
                file_name=selected_path.name,
                mime="text/csv",
                key=f"download_design_{selected_label}",
            )


def render_layers() -> None:
    """Render the dataset explorer for every system layer.

    Returns:
        None.
    """

    st.subheader("Datasets de las capas")
    layers = layer_summary()
    st.dataframe(layers, use_container_width=True, hide_index=True)
    layer_tabs = st.tabs(list(LAYER_MODULES))
    for tab, (layer_name, layer_module) in zip(layer_tabs, LAYER_MODULES.items()):
        with tab:
            config = LAYER_CATALOG[layer_name]
            folder = config["folder"]
            main_path = folder / config["main"]
            df = layer_module.load_dataset()
            st.markdown(f"**Rol sistemico:** {config['role']}")
            st.caption(rel(main_path))
            if df.empty:
                render_missing_file(main_path)
                continue

            sub_summary, sub_data, sub_visual, sub_docs = st.tabs(["Resumen", "Datos", "Visual", "Soporte"])
            with sub_summary:
                render_dataset_metrics(df)
                key_variables = layer_module.available_key_variables()
                if key_variables:
                    st.markdown("**Variables clave detectadas**")
                    st.write(", ".join(key_variables))
                else:
                    st.info("No se detectaron variables clave declaradas para esta capa.")
                render_missing_profile(df)

            with sub_data:
                compact = layer_module.feature_frame()
                data_tab, compact_tab = st.tabs(["Dataset completo", "Vista compacta"])
                with data_tab:
                    st.dataframe(df.head(500), use_container_width=True, hide_index=True)
                with compact_tab:
                    st.dataframe(compact.head(500), use_container_width=True, hide_index=True)
                st.download_button(
                    f"Descargar muestra {layer_name}",
                    data=df.head(500).to_csv(index=False).encode("utf-8"),
                    file_name=f"{layer_name.replace(' ', '_').replace('-', '').lower()}_muestra.csv",
                    mime="text/csv",
                    key=f"download_{layer_name}",
                )

            with sub_visual:
                render_time_series(df, layer_name)
                render_numeric_overview(df, f"layer_{layer_name}")
                render_layer_images(folder)

            with sub_docs:
                for label, file_name in SUPPORT_FILES.items():
                    support_path = folder / file_name
                    support_df = load_excel(support_path)
                    with st.expander(label, expanded=label == "Metadata"):
                        if support_df.empty:
                            render_missing_file(support_path)
                        else:
                            st.caption(rel(support_path))
                            st.dataframe(support_df.head(300), use_container_width=True, hide_index=True)
                readme = read_text(folder / "08_README.md")
                if readme:
                    with st.expander("README de la capa"):
                        st.markdown(readme)


def render_master_dataset() -> None:
    """Render the master dataset explorer.

    Returns:
        None.
    """

    st.subheader("Explorador del dataset maestro")
    selected_file = st.selectbox("Archivo maestro", list(MASTER_FILES))
    selected_path = MASTER_FILES[selected_file]
    dataset = load_csv(selected_path)
    if dataset.empty:
        render_missing_file(selected_path)
        return

    tab_general, tab_series, tab_groups, tab_quality, tab_table = st.tabs(
        ["General", "Series", "Grupos", "Cobertura", "Tabla"]
    )
    with tab_general:
        render_dataset_metrics(dataset)
        st.caption(rel(selected_path))
        date_col = find_date_column(dataset)
        if date_col:
            dates = pd.to_datetime(dataset[date_col], errors="coerce")
            start_date = dates.min()
            end_date = dates.max()
            c1, c2 = st.columns(2)
            c1.metric("Inicio", format_metric_date(start_date))
            c2.metric("Fin", format_metric_date(end_date))
    with tab_series:
        render_time_series(dataset, "master")
    with tab_groups:
        available_groups = {
            group: [col for col in cols if col in dataset.columns]
            for group, cols in FEATURE_GROUPS.items()
        }
        group = st.selectbox("Grupo", [name for name, cols in available_groups.items() if cols])
        cols = available_groups[group]
        render_numeric_overview(dataset[cols], f"master_group_{group}")
    with tab_quality:
        render_missing_profile(dataset)
        if selected_file != "Cobertura variables" and COVERAGE_PATH.exists():
            st.markdown("**Resumen de cobertura precomputado**")
            st.dataframe(load_csv(COVERAGE_PATH), use_container_width=True, hide_index=True)
    with tab_table:
        st.dataframe(dataset.head(1000), use_container_width=True, hide_index=True)
        st.download_button(
            "Descargar vista completa",
            data=dataset.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_file.replace(' ', '_').lower()}.csv",
            mime="text/csv",
        )


def render_notebooks() -> None:
    """Render the notebook inventory and documentation view.

    Returns:
        None.
    """

    st.subheader("Memoria metodologica en notebooks")
    st.write(
        "Esta carpeta conserva la exploracion y el paso a paso. La app y los "
        "datos quedan separados para que el repositorio funcione como producto reproducible."
    )
    notebooks = sorted(NOTEBOOKS_DIR.rglob("*.ipynb"))
    rows = []
    for notebook in notebooks:
        rows.append(
            {
                "Notebook": notebook.name,
                "Carpeta": rel(notebook.parent),
                "Tamano KB": round(notebook.stat().st_size / 1024, 1),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    md_files = sorted(NOTEBOOKS_DIR.rglob("*.md"))
    if md_files:
        selected_doc = st.selectbox("Documento", [rel(path) for path in md_files])
        st.markdown(read_text(BASE_DIR / selected_doc))
