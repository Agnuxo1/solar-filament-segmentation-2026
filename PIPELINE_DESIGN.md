# Solar Filament Segmentation Challenge 2026 — pipeline design

Estado: diseño inicial, 21-09-2026.

## Objetivo operativo

Maximizar Panoptic Quality (PQ) con máscaras de instancias válidas para el reto de Kaggle. La salida final es un CSV con `filament_id` y `segmentation_rle`; cada RLE representa una máscara de un único filamento en una imagen de 2048×2048.

La métrica obliga a optimizar simultáneamente:

\[
PQ = \frac{\sum_{(g,p)\in TP} IoU(g,p)}{|TP| + 0.5|FP| + 0.5|FN|}.
\]

Por tanto, una máscara muy bonita pero fragmentada pierde por FP/FN, y una máscara fusionada pierde correspondencias de instancias.

## Arquitectura propuesta: detector global + refinador morfológico

### 1. Ingesta y auditoría

* Leer cada JPEG como un solo canal `uint8`; no duplicar a RGB.
* Validar dimensiones, decodificación de polígonos/RLE, áreas y cajas COCO.
* Separar `physical_image_id` del `image_id` COCO: varios registros pueden apuntar al mismo JPEG por anotaciones de distintos anotadores.
* Parsear fecha e instrumento desde `YYYYMMDDHHMMSSII` y guardar metadatos solamente para particionar, nunca como entrada del modelo.

### 2. Etiquetas de entrenamiento

Crear dos vistas:

1. **Vista de instancias**: cada anotación COCO conserva su polígono y `spine` para entrenar la propuesta global.
2. **Vista de consenso**: para el mismo JPEG, emparejar filamentos de distintos anotadores mediante IoU de máscara, distancia entre spines y proximidad de cajas. Formar un objetivo blando por voto de anotadores; conservar como instancia la unión recortada del cluster cuando la concordancia sea suficiente. Las anotaciones no emparejables se mantienen como instancias separadas.

No rasterizar durante la conversión inicial: conservar polígonos y rasterizar sólo al crear batches o al escribir RLE. Esto evita pérdida de vértices finos.

### 3. Propuesta global

Fine-tunear un modelo de segmentación de instancias preentrenado, preferentemente YOLO-seg de tamaño pequeño/medio o equivalente, a 1024–1280 px. Entrada: imagen normalizada y, opcionalmente, dos canales deterministas derivados de ella:

* contraste local CLAHE;
* respuesta de ridge multiescala (Frangi/Sato o gradiente Hessiano).

La primera prueba debe ser imagen grayscale replicada a 3 canales para aislar el efecto de los canales auxiliares. Entrenamiento con `BCE/Focal + Dice` para máscara y pérdida de caja/objectness; rotaciones de 90°, flips, escala limitada y variaciones suaves de contraste/gamma. No usar color jitter.

El detector debe tener un umbral bajo durante la propuesta (`conf≈0.05–0.15`) y delegar la decisión final al refinador/calibrador.

### 4. Refinador por instancia

Para cada propuesta, extraer un crop cuadrado ampliado 1.4–1.8× alrededor de la caja y redimensionar a 384–512 px. Un U-Net/SegFormer ligero recibe:

* intensidad normalizada;
* contraste local;
* mapa de ridge;
* máscara de la propuesta global como cuarto canal.

Produce tres salidas:

* probabilidad de máscara;
* probabilidad de borde;
* probabilidad de spine/centroline.

Pérdida recomendada: `0.45 Dice/BCE + 0.25 boundary + 0.20 clDice/topology + 0.10 spine`. El spine no se envía a Kaggle, pero sirve para impedir que el refinador rompa filamentos largos y delgados.

El objetivo del refinador es la instancia emparejada, no la unión semántica de todos los filamentos del crop.

### 5. Descubrimiento de filamentos perdidos

Ejecutar una segunda pasada sobre tiles solapados de 1024–1280 px y sobre la imagen completa. Fusionar propuestas de ambas pasadas por IoU y distancia de spine. Sólo aceptar una detección nueva si supera el umbral calibrado de objectness/refiner y tiene soporte estructural (área mínima, longitud de spine o continuidad ridge). Esto recupera filamentos pequeños en la resolución global sin inundar PQ de FP.

### 6. Ensamblado y exclusividad

* Aplicar TTA de flips horizontal/vertical; añadir rotaciones sólo si la validación agrupada demuestra beneficio.
* Hacer matching de máscaras entre TTA/pasadas con IoU y distancia de centroline.
* Promediar logits alineados para duplicados; no unir dos filamentos distintos sólo porque sus cajas se solapan.
* Calcular `score = detector_conf × refiner_quality × continuity_quality`.
* Ordenar por score y hacer supresión suave/greedy. Ajustar los límites de score, área y número máximo de instancias por imagen únicamente con OOF.
* Convertir a máscaras mutuamente exclusivas: por orden de score, asignar cada píxel al candidato ganador; descartar máscaras cuyo área restante caiga por debajo del mínimo OOF.
* Antes de exportar, comprobar que ninguna pareja de instancias comparte píxeles.

## Validación

Usar 5 folds agrupados, sin separar entre train y validación:

* el mismo `physical_image_id`;
* idealmente el mismo mes calendario e instrumento si el tamaño lo permite, para medir generalización temporal/instrumental.

Calcular localmente el PQ oficial (matching IoU, TP/FP/FN), además de SQ, RQ, IoU/Dice por instancia, distribución de número de predicciones, falsos positivos, falsos negativos, casos one-to-many/many-to-one y porcentaje de imágenes vacías. El calibrador y los umbrales se ajustan con OOF, nunca con el fold usado para elegir pesos.

Experimentos mínimos, en este orden:

1. detector global grayscale + máscara directa;
2. + canales CLAHE/ridge;
3. + refinador de crop;
4. + spine/clDice;
5. + tiles para recall;
6. + TTA y calibración PQ.

Guardar por experimento checkpoint, configuración, semilla, fold, número de instancias y CSV de predicciones; no comparar sólo un score agregado.

## Exportación y auditoría

* Generar `filament_id` a partir del nombre físico sin extensión más sufijo `_1`, `_2`, ….
* Codificar con `pycocotools.mask.encode` en orden Fortran y escribir sólo `counts` ASCII, sin comillas.
* Decodificar cada RLE y comprobar forma 2048×2048, área positiva, round-trip exacto, IDs únicos y ausencia de solapamientos.
* Incluir también imágenes sin detecciones en la auditoría; no inventar filas de máscara vacía.
* Entregar notebook end-to-end, `requirements.txt` fijado y script de auditoría antes de cualquier submission.

## Riesgos y decisiones

* **Anotadores múltiples:** entrenar directamente todas las copias como si fueran observaciones independientes puede introducir duplicación y objetivos incompatibles. La vista de consenso debe compararse contra la vista cruda en OOF.
* **Resolución:** 2048² favorece inferencia por tiles y refinamiento; reducir toda la imagen a 512 px destruirá filamentos finos.
* **Fragmentación:** aumentar recall con umbral bajo sólo es útil si el refinador y la exclusividad reducen FP; PQ penaliza ambos extremos.
* **Generalización:** no usar fecha/instrumento como feature; sólo como grupos de validación y diagnóstico.
* **Reproducibilidad:** mantener separado el score público de Kaggle del score OOF y del score del self-evaluation oficial.

## Fuentes públicas

* [Kaggle — Overview/announcements](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/announcements)
* [Kaggle — Data](https://www.kaggle.com/competitions/filament-segmentation-2026/data)
* [Kaggle — leaderboard ranking](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/leaderboard-ranking)
* [Kaggle — self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook)
