# DATA_APIX

# 🧠 Verificación Biométrica Facial con InsightFace

Este repositorio contiene un pipeline completo para la **verificación de identidad facial**, basado en el uso de embeddings faciales generados con la librería `InsightFace`. El sistema permite comparar dos imágenes y determinar si corresponden a la misma persona utilizando medidas de similitud.



## 📁 Estructura del Repositorio

```
.
├── embeddings.py              # Extracción de embeddings faciales
├── verificacion.py           # Cálculo de similitud y evaluación de match
├── preprocesamiento.py       # Utilidades de carga de imágenes (RGB)
├── test_embeddings.py        # Prueba de extracción de embeddings
├── test_verificacion.py      # Prueba completa de verificación facial

```


## 🚀 Cómo funciona

1. **Carga de imágenes:** se leen dos imágenes desde rutas locales.
2. **Extracción de embeddings:** se utiliza `InsightFace` para detectar el rostro y obtener un vector numérico.
3. **Comparación:** se calcula la similitud coseno entre ambos vectores.
4. **Evaluación:** se define si hay "match" o no con base en un umbral.



## 🔧 Requisitos

- Python 3.9+
- OpenCV
- NumPy
- InsightFace
- (Opcional) TQDM, Matplotlib para debug visual

Instalación rápida:
```bash
pip install opencv-python-headless numpy insightface tqdm matplotlib
```



## 🧪 Uso de prueba

### Extraer embeddings:
```bash
python test_embeddings.py
```

### Verificar identidad:
```bash
python test_verificacion.py
```



## 📏 Parámetros de verificación

- Similitud coseno (`similitud_coseno`)
- Umbral configurable (por defecto: 0.35)
- Resultado binario: "MATCH" o "NO MATCH"



## 🧠 Casos de uso

- Validación de identidad con cédulas
- Comparación de imágenes biométricas
- Verificación antifraude (foto vs documento)



## 📌 Nota

Este repositorio se encuentra en desarrollo activo. Próximas mejoras incluirán:
- Evaluación por lotes
- Métricas de rendimiento
- Interfaz gráfica o API de servicio
