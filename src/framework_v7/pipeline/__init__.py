"""Reusable pipeline modules extracted from notebooks C01-C16.

The notebooks remain as methodological memory. This package contains the
functions that should be reused when a notebook, script or app needs to execute
the same business logic again.
"""

from __future__ import annotations

from types import ModuleType

from . import (
    domain_knowledge,
    evaluation,
    experiment_design,
    feature_engineering,
    integration,
    interpretation,
    ipml,
    layer_extraction,
    layer_framework,
    live_prediction,
    machine_learning,
    ml_preparation,
    model_governance,
    modeling,
    utils,
)


PIPELINE_MODULES = {
    "C01-C07 - Extraccion de capas": layer_extraction,
    "C01-C07 - Framework de capas": layer_framework,
    "C08 - Integracion": integration,
    "C09 - Ingenieria de datos": feature_engineering,
    "C10 - Catalogo de conocimiento": domain_knowledge,
    "C11 - IPML": ipml,
    "C12 - Preparacion ML": ml_preparation,
    "C13 - Machine Learning": machine_learning,
    "C14 - Modelado": modeling,
    "C15 - Evaluacion": evaluation,
    "C16 - Interpretacion": interpretation,
    "C17 - Gobierno de modelos": model_governance,
    "Diseno experimental": experiment_design,
    "Prediccion live": live_prediction,
    "Utilidades": utils,
}


def get_pipeline_module(stage_name: str) -> ModuleType:
    """Return the module that implements a registered notebook stage.

    Args:
        stage_name (str): Human-readable stage name registered in
            ``PIPELINE_MODULES``.

    Returns:
        ModuleType: Python module implementing the requested notebook stage.

    Raises:
        KeyError: If the stage is not registered.
    """

    return PIPELINE_MODULES[stage_name]
