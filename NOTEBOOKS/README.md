# NOTEBOOKS

Esta carpeta contiene los workflows de preparacion, modelado y analisis que
alimentan la aplicacion FRAMEWORK V7. Los notebooks funcionan como entorno de
trabajo para generar artefactos, mientras que la app consume los resultados
consolidados desde `DATA` y las funciones reutilizables desde `src`.

## Organizacion

- `C01` a `C07`: extraccion y estandarizacion de capas.
- `C08` y `C09`: integracion y preparacion del dataset maestro.
- `C10` y `C11`: catalogo de conocimiento e indicadores.
- `C12` a `C16`: preparacion de machine learning, modelado, evaluacion e
  interpretacion de resultados.
- `C17_GOBIERNO_MODELOS`: trazabilidad, versionamiento, riesgos y controles de
  modelos.
- `DISENO_EXPERIMENTAL`: configuracion de experimentos y variables objetivo.

## Uso Recomendado

1. Ejecutar notebooks solo cuando sea necesario regenerar artefactos.
2. Guardar salidas consolidadas en `DATA`.
3. Usar `app.py` para consultar resultados y operar el tablero.
4. Reutilizar funciones desde `src/framework_v7/pipeline` para evitar logica
   duplicada.

## Modulos Asociados

- `C08_INTEGRACION` usa `pipeline/integration.py`.
- `C09_INGENIERIA_DATOS` usa `pipeline/feature_engineering.py`.
- `C10_CKD` usa `pipeline/domain_knowledge.py`.
- `C11_IPML` usa `pipeline/ipml.py`.
- `C12_PREPARACION_MACHINE_LEARNING` usa `pipeline/ml_preparation.py`.
- `C13_MACHINE_LEARNING` usa `pipeline/machine_learning.py`.
- `C14_MODELADO` usa `pipeline/modeling.py`.
- `C15_EVALUACION` usa `pipeline/evaluation.py`.
- `C16_INTERPRETACION_RESULTADOS` usa `pipeline/interpretation.py`.
- `C17_GOBIERNO_MODELOS` usa `pipeline/model_governance.py`.
- `DISENO_EXPERIMENTAL` usa `pipeline/experiment_design.py`.
- `C01_CLIMATICA` a `C07_GOBERNANZA` usan `pipeline/layer_extraction.py` para
  inventario y perfil de calidad.
- `FW7_C01_Framework` a `FW7_C07_Framework` usan
  `pipeline/layer_framework.py` para metadata, diccionario, auditoria,
  indicadores y exportacion de artefactos.

Patron recomendado dentro de cada notebook:

```python
from framework_v7.pipeline.feature_engineering import build_engineered_master
from framework_v7.pipeline.layer_extraction import layer_execution_summary
from framework_v7.pipeline.layer_framework import build_layer_framework_artifacts
from framework_v7.pipeline.machine_learning import create_temporal_sequences
from framework_v7.pipeline.model_governance import governance_summary
```

## Validacion

Para validar que los notebooks y modulos siguen alineados con la app:

```bash
set PYTHONPATH=src
python -m framework_v7.pipeline.main
```

El archivo `src/framework_v7/pipeline/utils.py` contiene funciones de apoyo
para lectura, escritura, metadata, inventarios y validacion de columnas.
