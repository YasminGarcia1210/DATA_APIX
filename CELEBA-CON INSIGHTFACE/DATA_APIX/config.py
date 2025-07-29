# Configuraci�n del Pipeline de Verificaci�n Facial

# Modelos disponibles de InsightFace
MODELS = {
    'buffalo_l': 'Modelo recomendado para uso general',
    'arcface_r50': 'Modelo alternativo con mayor precisi�n',
    'antelopev2': 'Modelo m�s reciente (requiere m�s recursos)'
}

# Configuraciones de procesamiento
FACE_DETECTION_SIZE = (640, 640)
EMBEDDING_THRESHOLD = 0.4
BATCH_SIZE = 32

# Rutas de datos
DATA_DIR = "data/celeba"
EMBEDDINGS_DIR = "data/celeba/embeddings"
ALIGNED_FACES_DIR = "data/celeba/aligned_faces"

# URLs de descarga de CelebA
CELEBA_URLS = {
    'img_align_celeba': 'https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM',
    'identity_CelebA': 'https://drive.google.com/uc?id=1_ee_0u7MG9p55rDr96QmNk11GQIG8y35'
}

# Configuraciones de visualizaci�n
FIGURE_SIZE = (12, 8)
DPI = 300
