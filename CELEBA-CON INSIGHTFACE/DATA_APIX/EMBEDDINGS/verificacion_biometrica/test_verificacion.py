#!/usr/bin/env python3
"""
Test de verificación facial usando embeddings
Compara si dos imágenes pertenecen a la misma persona
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from celeba_pipeline import CelebAProcessor
from embedding_utils import compare_embeddings, cosine_similarity

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_faces(image_path1: str, image_path2: str, model_name: str = "buffalo_l", threshold: float = 0.5):
    """
    Compara dos imágenes y verifica si son de la misma persona

    Args:
        image_path1: Ruta de la primera imagen
        image_path2: Ruta de la segunda imagen
        model_name: Modelo de InsightFace
        threshold: Umbral de similitud para decidir verificación
    """
    logger.info("🔍 Iniciando verificación facial")
    processor = CelebAProcessor(model_name=model_name)

    # Alinear rostros
    face1 = processor.align_face(image_path1)
    face2 = processor.align_face(image_path2)

    if face1 is None or face2 is None:
        logger.error("❌ No se detectó rostro en una o ambas imágenes")
        return

    # Extraer embeddings
    emb1 = processor.extract_embedding(face1)
    emb2 = processor.extract_embedding(face2)

    if emb1 is None or emb2 is None:
        logger.error("❌ No se pudo extraer embeddings")
        return

    # Calcular similitud
    similarity = cosine_similarity(emb1, emb2)
    is_same = similarity >= threshold

    # Mostrar resultado
    logger.info(f"🧠 Similitud: {similarity:.4f} | Umbral: {threshold}")
    if is_same:
        logger.info("✅ Las imágenes corresponden a la misma persona")
    else:
        logger.info("❌ Las imágenes son de personas diferentes")

    # Retornar resultados
    return {
        "similarity": similarity,
        "threshold": threshold,
        "is_same_person": is_same,
        "image1": image_path1,
        "image2": image_path2
    }

def main():
    # Rutas de ejemplo (ajusta según tus imágenes)
    img1 = "data/sample_images/foto_1.jpg"
    img2 = "data/sample_images/foto_2.jpg"

    result = verify_faces(img1, img2, model_name="buffalo_l", threshold=0.55)

    if result:
        print("\n📋 RESULTADO:")
        print(f"Imagen 1: {result['image1']}")
        print(f"Imagen 2: {result['image2']}")
        print(f"Similitud: {result['similarity']:.4f}")
        print(f"¿Misma persona? {'✅ Sí' if result['is_same_person'] else '❌ No'}")

if __name__ == "__main__":
    main()
