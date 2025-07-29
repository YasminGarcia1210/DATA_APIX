#!/usr/bin/env python3
"""
Script de instalación y configuración del pipeline de verificación facial
Instala dependencias y configura el entorno para trabajar con CelebA e InsightFace
"""

import subprocess
import sys
import os
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    logger.info(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completado")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error en {description}: {e}")
        logger.error(f"Salida del error: {e.stderr}")
        return False

def check_python_version():
    logger.info("🐍 Verificando versión de Python...")
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        logger.error(f"❌ Se requiere Python 3.8+ (actual: {python_version.major}.{python_version.minor})")
        return False
    logger.info(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} detectado")
    return True

def install_requirements():
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        logger.error("❌ Archivo requirements.txt no encontrado")
        return False
    logger.info("📦 Instalando dependencias...")
    if not run_command(f'"{sys.executable}" -m pip install --upgrade pip', "Actualizando pip"):
        return False
    if not run_command(f'"{sys.executable}" -m pip install -r requirements.txt', "Instalando dependencias"):
        return False
    return True

def setup_directories():
    logger.info("📁 Creando estructura de directorios...")
    directories = [
        "data",
        "data/celeba",
        "data/celeba/img_align_celeba",
        "data/celeba/aligned_faces",
        "data/celeba/embeddings",
        "data/sample_images",
        "output",
        "models"
    ]
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"  📂 {directory}")
    logger.info("✅ Estructura de directorios creada")
    return True

def download_insightface_models():
    logger.info("🤖 Configurando modelos de InsightFace...")
    try:
        import insightface
        from insightface.app import FaceAnalysis
        logger.info("Descargando modelo buffalo_l...")
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("✅ Modelo buffalo_l configurado")
        try:
            logger.info("Descargando modelo arcface_r50...")
            app_arcface = FaceAnalysis(name='arcface_r50', providers=['CPUExecutionProvider'])
            app_arcface.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("✅ Modelo arcface_r50 configurado")
        except Exception as e:
            logger.warning(f"⚠️  No se pudo configurar arcface_r50: {e}")
        return True
    except Exception as e:
        logger.error(f"❌ Error configurando modelos InsightFace: {e}")
        return False

def create_sample_config():
    logger.info("⚙️  Creando archivo de configuración...")
    config_content = """# Configuración del Pipeline de Verificación Facial

# Modelos disponibles de InsightFace
MODELS = {
    'buffalo_l': 'Modelo recomendado para uso general',
    'arcface_r50': 'Modelo alternativo con mayor precisión',
    'antelopev2': 'Modelo más reciente (requiere más recursos)'
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

# Configuraciones de visualización
FIGURE_SIZE = (12, 8)
DPI = 300
"""
    config_file = Path("config.py")
    with open(config_file, 'w') as f:
        f.write(config_content)
    logger.info(f"✅ Archivo de configuración creado: {config_file}")
    return True

def create_download_script():
    logger.info("📥 Creando script de descarga de CelebA...")
    download_script = '''#!/usr/bin/env python3
"""
Script para descargar el dataset CelebA
"""

import gdown
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def download_celeba():
    """Descargar dataset CelebA desde Google Drive"""
    data_dir = Path("data/celeba")
    data_dir.mkdir(parents=True, exist_ok=True)
    files_to_download = {
        'img_align_celeba.zip': 'ID_DEL_ARCHIVO_DE_IMAGENES',
        'identity_CelebA.txt': 'ID_DEL_ARCHIVO_DE_IDENTIDADES'
    }
    print("📥 Para descargar CelebA:")
    print("1. Ve a: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
    print("2. Descarga 'Align&Cropped Images' e 'Identity CelebA'")
    print(f"3. Extrae los archivos en: {data_dir}")
    print("4. Ejecuta: python celeba_pipeline.py")

if __name__ == "__main__":
    download_celeba()
'''
    script_file = Path("download_celeba.py")
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(download_script)
    script_file.chmod(0o755)
    logger.info(f"✅ Script de descarga creado: {script_file}")
    return True

def run_tests():
    logger.info("🧪 Ejecutando pruebas básicas...")
    try:
        logger.info("Test 1: Importando dependencias...")
        import cv2
        import numpy as np
        import insightface
        import matplotlib.pyplot as plt
        import sklearn
        logger.info("✅ Dependencias importadas correctamente")
        logger.info("Test 2: Verificando OpenCV...")
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        assert test_img.shape == (100, 100, 3)
        logger.info("✅ OpenCV funcionando correctamente")
        logger.info("Test 3: Verificando InsightFace...")
        from insightface.app import FaceAnalysis
        logger.info("✅ InsightFace importado correctamente")
        logger.info("🎉 Todas las pruebas básicas pasaron")
        return True
    except Exception as e:
        logger.error(f"❌ Error en las pruebas: {e}")
        return False

def print_next_steps():
    logger.info("🎯 Configuración completada. Próximos pasos:")
    steps = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                          PRÓXIMOS PASOS                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ 1. 📥 DESCARGAR CELEBA (Opcional):                                          ║
║    • Ve a: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html                 ║
║    • Descarga "Align&Cropped Images" e "Identity CelebA"                    ║
║    • Extrae en: data/celeba/                                                 ║
║ 2. 🚀 EJECUTAR DEMO:                                                        ║
║    python demo_pipeline.py                                                  ║
║ 3. 🧠 PROCESAR CELEBA (si lo descargaste):                                  ║
║    python celeba_pipeline.py                                                ║
║ 4. 📊 ANALIZAR EMBEDDINGS:                                                  ║
║    python embedding_utils.py                                                ║
║ 5. 📚 EJEMPLOS DE USO:                                                      ║
║    • Ver los archivos de ejemplo en EMBEDDINGS/verificacion_biometrica/    ║
║    • Revisar la documentación en README.md                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(steps)

def main():
    logger.info("🚀 Iniciando configuración del pipeline de verificación facial")
    success = True
    if not check_python_version(): success = False
    if success and not setup_directories(): success = False
    if success and not install_requirements(): success = False
    if success and not download_insightface_models(): success = False
    if success and not create_sample_config(): success = False
    if success and not create_download_script(): success = False
    if success and not run_tests(): success = False
    if success:
        logger.info("🎉 ¡Configuración completada exitosamente!")
        print_next_steps()
    else:
        logger.error("❌ La configuración falló. Revisa los errores anteriores.")
        sys.exit(1)

if __name__ == "__main__":
    main()
