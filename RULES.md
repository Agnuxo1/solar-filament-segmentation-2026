# Solar Filament Segmentation Challenge 2026 — reglas verificadas

Última verificación: 2026-09-21. Fuentes públicas consultadas:

- https://www.kaggle.com/competitions/filament-segmentation-2026/rules
- https://www.kaggle.com/competitions/filament-segmentation-2026/overview/announcements
- https://www.kaggle.com/competitions/filament-segmentation-2026/overview/leaderboard-ranking
- https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook

## Reglas operativas

- Competición: Solar Filament Segmentation Challenge 2026; patrocinador: NSF National Solar Observatory.
- La métrica vigente es Panoptic Quality (PQ), con penalización explícita de falsos positivos y falsos negativos; no se optimizará la antigua métrica Dice como sustituta.
- La salida es un único CSV del conjunto de test, con `filament_id` y `segmentation_rle`; cada fila representa un filamento predicho.
- La inferencia debe utilizar únicamente las imágenes H-alpha del directorio de test. No se usarán metadatos ground-truth externos para entrenar modelos.
- Los datos externos y modelos externos son admisibles si son públicos, razonablemente accesibles y de coste mínimo para los participantes. Cualquier etiqueta ground-truth externa al reto queda excluida.
- Máximo: 5 submissions por día; se pueden seleccionar hasta 2 submissions finales.
- Solo se usará una cuenta Kaggle y un único equipo; el tamaño máximo del equipo es 5.
- No se usarán etiquetado manual ni predicciones humanas del conjunto de validación/test.
- La fecha publicada para entregar informe final y solución es el 15 de noviembre de 2026; el anuncio de ganadores indica el 30 de noviembre de 2026. Las fechas pueden cambiar en la web oficial.
- Para evaluación final se solicita un informe técnico de 4 páginas, un repositorio Git público con `requirements.txt` versionado y un notebook que reproduzca el pipeline.
- Si la solución resultara ganadora, el código debe poder entregarse bajo una licencia aprobada por OSI y ser reproducible; cualquier dependencia externa debe tener licencia compatible.
- Se permite usar IA/AutoML, pero el participante debe poder explicar, justificar y reproducir los componentes principales de la solución.

## Política de credenciales

No se almacenan tokens, contraseñas, API keys ni archivos de sesión en el repositorio. Las credenciales, si el usuario las configura por su cuenta, deben llegar únicamente mediante variables de entorno o un `kaggle.json` local fuera del control de versiones. No se imprimirán ni se copiarán desde documentos auxiliares.

## Estado de verificación

- Reglas públicas leídas y resumidas: sí.
- Dataset local descargado/verificado: sí; archivo Kaggle SHA-256 `56d8cd2927859fa7df72bd28ad1586f53c5814df6f81db02e755223e430657df`, 707 train JPEG, 180 test JPEG, todos 2048×2048 grayscale.
- Self-evaluation oficial replicada: sí; `scripts/evaluate_pq.py` reproduce el umbral IoU > 0.5 y el cálculo PQ del notebook público; pruebas sintéticas y evaluación real del baseline vacío completadas.
- Límite diario de submissions consultado antes de cualquier envío: obligatorio.
- Límite actual consultado: 0 submissions encontradas para la cuenta/equipo configurado.
- Decisión de JEV sobre la primera hipótesis: pendiente; JEV fue consultado dos veces desde `E:/Rescate-C-2026-09-21/Documents/JEV-Orchestrator` (al inicio y al cambiar a experimentación), pero Codex CLI agotó ambos timeouts de 30 s y devolvió `MODEL BACKENDS TEMPORARILY UNAVAILABLE`. No se inició entrenamiento ni submission.
