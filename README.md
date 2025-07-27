# DATA_APIX

# 🧠 Pipeline Completo de Verificación Facial con InsightFace y CelebA

Este repositorio contiene un **pipeline completo y robusto** para verificación de identidad facial usando **InsightFace** y el dataset **CelebA**. El sistema incluye alineación facial, extracción de embeddings, análisis estadístico y herramientas de visualización.

## ✨ Características Principales

🔧 **Alineación Facial Automática**: Detección y alineación de rostros usando InsightFace  
🧠 **Extracción de Embeddings**: Vectores faciales con modelos `buffalo_l` y `arcface_r50`  
📊 **Análisis Estadístico**: Métricas intra e inter-identidad para optimización de umbrales  
📈 **Visualizaciones**: Gráficos 2D con t-SNE/PCA y distribuciones de similitud  
🎯 **Verificación Robusta**: Múltiples métricas de similitud y umbrales adaptativos  
📦 **Dataset CelebA**: Soporte completo para procesamiento por lotes  

## 📁 Estructura del Repositorio

```
.
├── celeba_pipeline.py         # Pipeline principal para CelebA + InsightFace
├── embedding_utils.py         # Utilidades de análisis y verificación
├── demo_pipeline.py          # Demostración completa del pipeline
├── setup.py                  # Script de instalación y configuración
├── requirements.txt          # Dependencias del proyecto
├── config.py                 # Configuración del sistema (generado)
├── download_celeba.py        # Script para descargar CelebA (generado)
│
├── data/
│   ├── celeba/              # Dataset CelebA
│   │   ├── img_align_celeba/     # Imágenes originales
│   │   ├── aligned_faces/        # Rostros alineados
│   │   ├── embeddings/           # Embeddings guardados (.npy, .pkl)
│   │   └── identity_CelebA.txt   # Metadatos de identidades
│   └── sample_images/       # Imágenes de prueba
│
└── EMBEDDINGS/verificacion_biometrica/  # Sistema original
    ├── embeddings.py
    ├── verificacion.py
    └── test_*.py
```

## 🚀 Instalación Rápida

### 1. Configuración Automática
```bash
# Instalar dependencias y configurar entorno
python setup.py
```

### 2. Instalación Manual
```bash
# Instalar dependencias
pip install -r requirements.txt

# Crear directorios
mkdir -p data/celeba/{img_align_celeba,aligned_faces,embeddings}
```

## 🧪 Uso del Pipeline

### 1. 🎭 Demo Completo (Recomendado para empezar)
```bash
python demo_pipeline.py
```
Este script:
- Crea imágenes de muestra automáticamente
- Demuestra alineación facial
- Extrae y visualiza embeddings
- Realiza verificación facial paso a paso

### 2. 🗂️ Procesamiento de CelebA
```bash
# Descargar CelebA (opcional - ver instrucciones en download_celeba.py)
python download_celeba.py

# Procesar dataset completo
python celeba_pipeline.py
```

### 3. 📊 Análisis de Embeddings
```bash
# Analizar embeddings generados
python embedding_utils.py
```

## 🔧 Uso Programático

### Alineación y Extracción de Embeddings
```python
from celeba_pipeline import CelebAProcessor

# Inicializar procesador
processor = CelebAProcessor(
    model_name="buffalo_l",  # o "arcface_r50"
    data_dir="data/celeba"
)

# Alinear rostro
aligned_face = processor.align_face("path/to/image.jpg")

# Extraer embedding
embedding = processor.extract_embedding(aligned_face)
print(f"Embedding shape: {embedding.shape}")  # (512,)
```

### Verificación de Identidad
```python
from embedding_utils import EmbeddingAnalyzer

# Inicializar analizador
analyzer = EmbeddingAnalyzer("data/celeba/embeddings/celeba_embeddings_full.pkl")

# Verificar identidad
result = analyzer.verify_identity(
    embedding1, embedding2, 
    threshold=0.4, 
    metric="cosine"
)

print(f"Match: {result['is_match']}")
print(f"Similarity: {result['score']:.4f}")
```

### Procesamiento por Lotes
```python
# Procesar múltiples imágenes
results = processor.process_images(
    images_dir="data/celeba/img_align_celeba",
    max_images=1000,
    save_aligned=True,
    save_embeddings=True
)

print(f"Procesadas: {results['stats']['processed']} imágenes")
print(f"Identidades únicas: {results['stats']['unique_identities']}")
```

## 📊 Análisis y Métricas

### Estadísticas de Calidad
```python
# Análisis intra-identidad (variabilidad dentro de cada persona)
intra_stats = analyzer.calculate_intra_identity_statistics()

# Análisis inter-identidad (separabilidad entre personas)
inter_stats = analyzer.calculate_inter_identity_statistics()

# Evaluación de umbrales óptimos
threshold_eval = analyzer.evaluate_threshold()
best_threshold = threshold_eval['best_f1_threshold']
```

### Visualizaciones
```python
# Distribución de similitudes
analyzer.plot_similarity_distribution("similarity_dist.png")

# Embeddings en 2D con t-SNE
analyzer.visualize_embeddings_2d(method="tsne", save_path="embeddings_2d.png")
```

## 🎯 Configuración de Modelos

### Modelos Disponibles
- **`buffalo_l`**: Recomendado para uso general (balance precisión/velocidad)
- **`arcface_r50`**: Mayor precisión, más lento
- **`antelopev2`**: Modelo más reciente (requiere más recursos)

### Parámetros de Verificación
- **Umbral Coseno**: 0.4-0.6 (típico para verificación facial)
- **Detección**: 640x640 píxeles para mejor precisión
- **Embedding**: Vector de 512 dimensiones normalizado

## 📥 Dataset CelebA

### Descarga Manual
1. Ve a: [CelebA Official](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
2. Descarga:
   - "Align&Cropped Images" → `data/celeba/img_align_celeba/`
   - "Identity CelebA" → `data/celeba/identity_CelebA.txt`

### Estructura Esperada
```
data/celeba/
├── img_align_celeba/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
└── identity_CelebA.txt
```

## 🔍 Casos de Uso

✅ **Verificación 1:1**: ¿Son la misma persona?  
🔎 **Búsqueda 1:N**: Encontrar identidad en base de datos  
📱 **Autenticación Móvil**: Verificación en dispositivos  
🏢 **Control de Acceso**: Sistemas de seguridad  
🆔 **Validación de Documentos**: Comparar foto vs cédula  

## 📈 Rendimiento Esperado

| Métrica | Valor Típico |
|---------|--------------|
| **Precisión** | 95-99% (con umbral optimizado) |
| **Velocidad** | ~50ms por imagen (CPU) |
| **Tamaño Embedding** | 512 dimensiones |
| **Memoria** | ~2KB por embedding |

## 🛠️ Personalización

### Cambiar Modelo
```python
processor = CelebAProcessor(model_name="arcface_r50")
```

### Ajustar Detección
```python
# En config.py
FACE_DETECTION_SIZE = (640, 640)  # Mayor resolución = más precisión
```

### Umbral Personalizado
```python
result = analyzer.verify_identity(emb1, emb2, threshold=0.5)
```

## 🤝 Contribuciones

Las contribuciones son bienvenidas! Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 🆘 Soporte

- 📖 **Documentación**: Lee los comentarios en el código
- 🐛 **Issues**: Reporta problemas en GitHub Issues
- 💬 **Discusiones**: Para preguntas generales

---

**⭐ Si este proyecto te es útil, considera darle una estrella!**
