# FRAMEWORK V7

Framework para la aplicacion de tecnologias 4.0 y ciencia de datos en la
gestion hidrica del Rio Bogota. El repositorio organiza datos biofisicos,
hidraulicos, climaticos, sociales e institucionales en un flujo multicapa para
explorar informacion, disenar experimentos y evaluar modelos predictivos.

## Objetivo

El proyecto convierte notebooks de investigacion en una base reproducible:

- `DATA`: datos de entrada, datos maestros, tensores, metricas y resultados.
- `NOTEBOOKS`: memoria metodologica y trazabilidad de los colabs.
- `src/framework_v7/layers`: funciones por capa del framework.
- `src/framework_v7/pipeline`: funciones reutilizables extraidas de notebooks.
- `main.py`: chequeo ligero de ejecucion por consola.
- `app.py`: tablero Streamlit para visualizar resultados.

## Anatomia Del Negocio

El flujo sigue una anatomia por capas y etapas:

1. Las capas C01-C07 consolidan informacion climatica, hidrologica, calidad de
   agua, ONI, hidraulica, percepcion y gobernanza.
2. C08-C09 integran las capas, preparan llaves `Fecha`/`Nodo`, imputan datos y
   generan el dataset maestro.
3. C10-C12 documentan conocimiento de dominio, pertinencia para machine
   learning y seleccion de variables.
4. C13 prepara datasets transformados y secuencias temporales.
5. C14 entrena o registra modelos y tensores por experimento.
6. C15 exporta predicciones y metadata de evaluacion.
7. C16 interpreta resultados y traduce metricas en lectura sistemica.
8. C17 documenta gobierno de modelos: versionamiento, trazabilidad, riesgos,
   controles y uso seguro de los modelos.

## Estructura

```text
FRAMEWORK_V7/
|-- DATA/
|   |-- RAW/
|   |-- MASTER/
|   |-- MACHINE_LEARNING/
|   |-- MODELADO/
|   |-- EVALUACIONES/
|   |-- INTERPRETACION_RESULTADOS/
|   `-- DISENO_EXPERIMENTAL/
|-- NOTEBOOKS/
|-- src/
|   `-- framework_v7/
|       |-- layers/
|       `-- pipeline/
|-- app.py
|-- main.py
|-- pyproject.toml
`-- requirements.txt
```

## Pipeline Modular De Notebooks

Los notebooks se conservan como evidencia, pero la logica reutilizable esta en
`src/framework_v7/pipeline`. Esta separacion permite importar funciones desde
Colab, scripts o pruebas sin repetir celdas largas.

- `layer_extraction.py`: inventario, trazabilidad y resumen de calidad de los
  notebooks C01-C07 y sus artefactos RAW/MASTER.
- `layer_framework.py`: gobierno del dato, hashes, metadata, diccionario,
  auditoria, indicadores, EDA y exportacion para los notebooks
  `FW7_C0X_Framework`.
- `utils.py`: lectura, escritura, validacion, metadata e inventario de
  artefactos.
- `integration.py`: integracion C08 y llaves `Fecha`/`Nodo`.
- `feature_engineering.py`: preparacion C09, temporalidad, imputacion y
  cobertura.
- `domain_knowledge.py`: catalogo C10.
- `ipml.py`: indice de pertinencia para machine learning C11.
- `ml_preparation.py`: seleccion de predictoras y variable objetivo C12.
- `machine_learning.py`: transformaciones, diagnosticos y secuencias C13.
- `modeling.py`: configuracion, tensores, registros y diagnosticos C14.
- `evaluation.py`: metricas, predicciones y recomendaciones C15.
- `interpretation.py`: resumenes, interpretacion y lectura sistemica C16.
- `model_governance.py`: gobierno de modelos C17, trazabilidad, matriz de
  riesgos y evidencia de ciclo de vida.
- `experiment_design.py`: diseno experimental y plan de experimentos.
- `main.py`: validacion ligera del pipeline modular.

Ejemplo desde un notebook:

```python
from framework_v7.pipeline.machine_learning import summarize_ml_experiments
from framework_v7.pipeline.interpretation import summarize_interpretation_experiments
from framework_v7.pipeline.layer_framework import build_layer_framework_artifacts
```

### Modularizacion C01-C07

Los notebooks de framework por capa conservan la trazabilidad metodologica, pero
las funciones repetidas ahora viven en `layer_framework.py`:

- `generate_record_hash`: reemplaza la funcion repetida `generar_hash`.
- `add_framework_governance_columns`: agrega columnas de gobierno del dato.
- `build_layer_metadata`: genera la estructura de `03_Metadata.xlsx`.
- `build_data_dictionary`: genera la estructura de `04_Diccionario_Datos.xlsx`.
- `audit_layer_dataset`: resume registros, variables, nulos y duplicados.
- `quality_indicators`: calcula completitud, nulos, duplicidad y cobertura.
- `build_layer_framework_artifacts`: produce los artefactos estandarizados de
  una capa en memoria.
- `export_layer_framework_artifacts`: exporta esos artefactos fuera de la app.

Ejemplo minimo desde un notebook C01-C07:

```python
from framework_v7.pipeline.layer_framework import build_layer_framework_artifacts

artifacts = build_layer_framework_artifacts(
    df,
    layer_name="C01 - Climatica",
    layer_code="C01",
    version="1.0",
    responsible="Jose Barreto y Juan Riataga",
    date_column="Fecha",
    node_column="Nodo",
)
```

## Experimentos Disponibles

El repositorio mantiene resultados versionados por carpeta de experimento:

- `Exp01`: experimento base de clasificacion sobre `irca`.
- `Exp01-V3`: nueva version de clasificacion sobre `irca`.
- `Exp04`: experimento de regresion sobre `VolumenUtilDiarioMasa`.

Los artefactos principales se encuentran en:

- `DATA/MACHINE_LEARNING/C13_MACHINE_LEARNING/Transformaciones/<Experimento>/`
- `DATA/MODELADO/Tensores/<Experimento>/`
- `DATA/MODELADO/Modelos/<Experimento>/`
- `DATA/MODELADO/Diagnosticos/<Experimento>/`
- `DATA/EVALUACIONES/<Experimento>/`
- `DATA/INTERPRETACION_RESULTADOS/<Experimento>/`

## Gobierno De Modelos

El notebook `NOTEBOOKS/C17_GOBIERNO_MODELOS/FW7_C17_Gobierno_modelos.ipynb`
consolida la memoria de gobierno para los modelos del framework. La app expone
esta informacion en la vista `Gobierno de modelos`, incluyendo:

- Modelos gobernados: `Exp01-V3` para `irca` y `Exp04` para
  `VolumenUtilDiarioMasa`.
- Identificacion y versionamiento del modelo de referencia.
- Trazabilidad de artefactos entre C09-C17.
- Arquitectura registrada del modelo LSTM.
- Estado de evidencias, riesgos, controles y matriz de uso seguro.

## Diseno Experimental

`DATA/DISENO_EXPERIMENTAL` define la planeacion de experimentos del framework:

- `catalogo_experimentos.csv`
- `configuracion_experimentos.csv`
- `variables_predictoras.csv`
- `estado_experimentos.csv`
- `criterios_clasificacion.csv`
- `criterios_regresion.csv`

La etapa de diseno conecta preguntas de investigacion, variables objetivo,
tipo de problema, ventana temporal, horizonte predictivo y modelo.

## Ejecucion Por Consola

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Validar el proyecto completo:

```bash
python main.py
```

Validar solo el pipeline modular de notebooks:

```bash
set PYTHONPATH=src
python -m framework_v7.pipeline.main
```

Instalar como paquete local para importar desde notebooks:

```bash
pip install -e .
```

## Aplicacion Streamlit

La app permite consultar los resultados sin ejecutar los notebooks:

```bash
streamlit run app.py
```

La modularizacion de notebooks vive en `src/framework_v7/pipeline` y es
independiente de `app.py`.

## Convenciones De Desarrollo

- Mantener notebooks como memoria metodologica.
- Mover funciones reutilizables a `src/framework_v7/pipeline`.
- Guardar resultados consolidados en `DATA`.
- Evitar logica pesada dentro de `app.py`.
- Ejecutar `python main.py` antes de publicar cambios.
- Documentar funciones con docstrings de estilo Google o Numpy.

## Autores

Proyecto desarrollado por Jose Barreto y Juan Riataga.
