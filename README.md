# FRAMEWORK V7

FRAMEWORK V7 es una aplicacion de analitica hidrica para explorar datos
multicapa del Rio Bogota, consultar resultados de modelos predictivos y simular
escenarios de riesgo o disponibilidad de agua desde un tablero Streamlit.

La solucion combina datos climaticos, hidrologicos, hidraulicos, calidad del
agua, percepcion y gobernanza en una estructura reproducible con `DATA`, `src`,
`NOTEBOOKS`, `main.py` y `app.py`.

## Que Incluye

- Dashboard Streamlit para explorar capas, datasets maestros y experimentos.
- Simulador de prediccion live para `irca` y `VolumenUtilDiarioMasa`.
- Vista de gobierno de modelos con trazabilidad, riesgos, controles y ciclo de
  vida.
- Pipeline modular en `src/framework_v7/pipeline`.
- Modulos por capa en `src/framework_v7/layers`.
- Artefactos versionados de datos, tensores, modelos, metricas y predicciones.

## Flujo De Datos

1. Las capas C01-C07 consolidan informacion climatica, hidrologica, calidad de
   agua, ONI, hidraulica, percepcion y gobernanza.
2. C08-C09 integran las capas, preparan llaves `Fecha`/`Nodo`, imputan datos y
   generan el dataset maestro.
3. C10-C12 preparan catalogos de conocimiento, pertinencia para machine
   learning y seleccion de variables.
4. C13 genera datasets transformados y secuencias temporales.
5. C14 registra tensores, modelos y salidas de entrenamiento.
6. C15 consolida predicciones y metricas.
7. C16 transforma resultados en lectura tecnica y sistemica.
8. C17 organiza gobierno de modelos: versionamiento, trazabilidad, riesgos,
   controles y uso seguro.

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

## Aplicacion Streamlit

Ejecutar localmente:

```bash
streamlit run app.py
```

Vistas principales:

- `Dashboard`: resumen ejecutivo, mapa del sistema, predicciones y calidad de
  datos.
- `Experimentos`: resultados, model cards, metricas, predicciones e
  interpretacion.
- `Prediccion live`: simulador interactivo y carga de CSV/Excel para inferencia
  exploratoria.
- `Gobierno de modelos`: modelos gobernados, trazabilidad, riesgos, controles y
  arquitectura.
- `Diseno experimental`: catalogo, configuracion y estado de experimentos.
- `Datasets por capas`: exploracion de fuentes por capa.
- `Dataset maestro`: perfil, cobertura, series, nulos y tabla final.
- `Notebooks`: inventario de workflows disponibles.

## Pipeline Modular

La logica reutilizable vive en `src/framework_v7/pipeline` para que la app,
scripts y notebooks usen las mismas funciones.

- `layer_extraction.py`: inventario y perfil de calidad de artefactos por capa.
- `layer_framework.py`: gobierno del dato, hashes, metadata, diccionario,
  auditoria, indicadores y exportacion de artefactos por capa.
- `utils.py`: lectura, escritura, validacion, metadata e inventario.
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
- `model_governance.py`: gobierno de modelos C17, trazabilidad y matriz de
  riesgos.
- `experiment_design.py`: catalogo y configuracion de experimentos.
- `main.py`: validacion ligera del pipeline modular.

Ejemplo:

```python
from framework_v7.pipeline.machine_learning import summarize_ml_experiments
from framework_v7.pipeline.model_governance import governance_summary
from framework_v7.pipeline.layer_framework import build_layer_framework_artifacts
```

## Experimentos Disponibles

- `Exp01`: clasificacion base sobre `irca`.
- `Exp01-V3`: version ajustada de clasificacion sobre `irca`.
- `Exp04`: regresion sobre `VolumenUtilDiarioMasa`.

Artefactos por experimento:

- `DATA/MACHINE_LEARNING/C13_MACHINE_LEARNING/Transformaciones/<Experimento>/`
- `DATA/MODELADO/Tensores/<Experimento>/`
- `DATA/MODELADO/Modelos/<Experimento>/`
- `DATA/MODELADO/Diagnosticos/<Experimento>/`
- `DATA/EVALUACIONES/<Experimento>/`
- `DATA/INTERPRETACION_RESULTADOS/<Experimento>/`

## Gobierno De Modelos

La vista `Gobierno de modelos` presenta el control operativo de los modelos:

- Modelos gobernados: `Exp01-V3` para `irca` y `Exp04` para
  `VolumenUtilDiarioMasa`.
- Identificacion y versionamiento del modelo de referencia.
- Trazabilidad entre datos, transformaciones, tensores, modelos, predicciones e
  interpretacion.
- Arquitectura registrada del modelo LSTM.
- Riesgos, controles, estado de seguimiento y matriz de uso seguro.

## Prediccion Live

La app incluye un simulador interactivo basado en un modelo liviano entrenado al
vuelo sobre el dataset maestro. Permite:

- Ajustar variables por capa mediante controles en pantalla.
- Comparar el escenario editado contra la mediana historica.
- Consultar variables con mayor influencia en el baseline.
- Cargar archivos CSV/Excel y descargar predicciones generadas.

Este simulador es una herramienta exploratoria para analisis de escenarios. Los
modelos entrenados y sus artefactos siguen disponibles en `DATA/MODELADO`.

## Ejecucion Por Consola

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Validar el proyecto:

```bash
python main.py
```

Validar solo el pipeline modular:

```bash
set PYTHONPATH=src
python -m framework_v7.pipeline.main
```

Instalar como paquete local:

```bash
pip install -e .
```

## Convenciones De Desarrollo

- Mantener `app.py` como punto de entrada de Streamlit.
- Centralizar logica reusable en `src/framework_v7/pipeline`.
- Guardar datasets y resultados consolidados en `DATA`.
- Mantener rutas canonicas en `src/framework_v7/paths.py`.
- Ejecutar `python main.py` antes de publicar cambios.
- Documentar funciones con docstrings de estilo Google o Numpy.

## Autores

Proyecto desarrollado por Jose Barreto y Juan Riataga.
