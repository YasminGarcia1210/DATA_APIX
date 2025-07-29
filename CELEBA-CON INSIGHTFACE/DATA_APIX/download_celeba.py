#!/usr/bin/env python3
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
