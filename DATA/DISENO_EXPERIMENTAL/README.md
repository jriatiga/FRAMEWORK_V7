# DISENO EXPERIMENTAL

Esta carpeta contiene la configuracion que usa FRAMEWORK V7 para registrar,
comparar y consultar experimentos predictivos desde notebooks, scripts y la app
Streamlit.

## Proposito

El diseno experimental conecta variables objetivo, variables predictoras, tipo
de problema, ventana temporal, horizonte predictivo, modelo y criterios de
seguimiento. Los archivos de esta carpeta son consumidos por los notebooks C13,
C14, C15 y C16, por el pipeline modular y por la vista `Diseno experimental` de
la aplicacion.

## Artefactos

- `catalogo_experimentos.csv`: catalogo de `Exp01` a `Exp08`.
- `configuracion_experimentos.csv`: parametros generales de modelado.
- `variables_predictoras.csv`: variables iniciales usadas como entrada.
- `estado_experimentos.csv`: avance y resultados por experimento ejecutado.
- `criterios_clasificacion.csv`: criterios para leer modelos de clasificacion.
- `criterios_regresion.csv`: criterios para leer modelos de regresion.

## Experimentos Ejecutados

| Experimento | Tipo | Variable objetivo | Estado | Resultado principal |
|---|---|---|---|---|
| Exp01 | Clasificacion | `irca` | Ejecutado | Linea base de clasificacion; accuracy 0.8462, precision 0.0000, recall 0.0000 y F1 0.0000. |
| Exp01-V3 | Clasificacion | `irca` | Ejecutado | Version ajustada para IRCA; accuracy 0.9226, precision 0.5645, recall 0.7292 y F1 0.6364. |
| Exp04 | Regresion | `VolumenUtilDiarioMasa` | Ejecutado | Modelo de volumen util; MAE 0.0824, RMSE 0.1470, MAPE 23.8944 y R2 0.7225. |

## Uso En La App

La vista `Diseno experimental` permite:

- Consultar catalogo, configuracion y estado de experimentos.
- Revisar variables predictoras disponibles.
- Comparar criterios de clasificacion y regresion.
- Descargar datasets de configuracion en formato CSV.

Los experimentos pendientes (`Exp02`, `Exp03`, `Exp05`, `Exp06`, `Exp07` y
`Exp08`) quedan disponibles para futuras iteraciones del producto.
