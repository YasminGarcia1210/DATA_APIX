#!/usr/bin/env python3
"""
Pipeline completo para procesamiento de CelebA con InsightFace
Incluye descarga, alineación facial y extracción de embeddings
"""

import os
import cv2
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import insightface
from insightface.app import FaceAnalysis
import requests
import zipfile
import gdown
from PIL import Image
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CelebAProcessor:
    """Procesador principal para el dataset CelebA con InsightFace"""

    def __init__(self, 
                 data_dir: str = "data/celeba",
                 model_name: str = "buffalo_l",
                 device: str = "cpu"):
        """
        Inicializar el procesador de CelebA

        Args:
            data_dir: Directorio donde guardar/cargar CelebA
            model_name: Modelo de InsightFace a usar (buffalo_l, arcface_r50, etc.)
            device: Dispositivo a usar (cpu, cuda)
        """
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.device = device

        # Crear directorios necesarios
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.aligned_dir = self.data_dir / "aligned_faces"
        self.embeddings_dir = self.data_dir / "embeddings"
        self.aligned_dir.mkdir(exist_ok=True)
        self.embeddings_dir.mkdir(exist_ok=True)

        # Inicializar modelo InsightFace
        self.face_app = None
        self.load_face_model()

        # Metadatos de CelebA
        self.identities = {}
        self.image_paths = []

    def load_face_model(self):
        """Cargar el modelo de InsightFace"""
        try:
            logger.info(f"Cargando modelo InsightFace: {self.model_name}")
            self.face_app = FaceAnalysis(name=self.model_name, providers=['CPUExecutionProvider'])
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("Modelo InsightFace cargado exitosamente")
        except Exception as e:
            logger.error(f"Error cargando modelo InsightFace: {e}")
            raise

    def download_celeba_sample(self, num_images: int = 1000):
        """
        Descargar una muestra del dataset CelebA
        Para el dataset completo, usar los enlaces oficiales
        """
        logger.info(f"Descargando muestra de CelebA ({num_images} imágenes)")

        # URLs de muestra (puedes cambiar por el dataset completo)
        sample_url = "https://drive.google.com/uc?id=1fJAF8xk0QzV2N6d_JgLZsYDH6ACHo0wJ"  # Ejemplo

        zip_path = self.data_dir / "celeba_sample.zip"
        images_dir = self.data_dir / "img_align_celeba"

        if not images_dir.exists():
            logger.info("Descargando dataset CelebA...")
            images_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directorio creado: {images_dir}")
            logger.info("⚠️  Por favor, descarga manualmente el dataset CelebA desde:")
            logger.info("   http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
            logger.info(f"   Y extrae las imágenes en: {images_dir}")
        else:
            logger.info(f"Dataset CelebA encontrado en: {images_dir}")

        return images_dir

    def load_celeba_identities(self):
        """Cargar información de identidades desde identity_CelebA.txt"""
        identity_file = self.data_dir / "identity_CelebA.txt"

        if not identity_file.exists():
            logger.warning(f"Archivo de identidades no encontrado: {identity_file}")
            logger.info("⚠️  Descarga identity_CelebA.txt desde el sitio oficial de CelebA")
            return {}

        identities = {}
        with open(identity_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    image_name = parts[0]
                    identity_id = int(parts[1])
                    identities[image_name] = identity_id

        logger.info(f"Cargadas {len(identities)} identidades de CelebA")
        return identities

    def align_face(self, image_path: str) -> Optional[np.ndarray]:
        """
        Alinear rostro usando InsightFace

        Args:
            image_path: Ruta de la imagen

        Returns:
            Imagen alineada o None si no se detecta rostro
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None

            faces = self.face_app.get(img)

            if len(faces) == 0:
                return None

            face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            aligned_face = face.norm_crop

            return aligned_face

        except Exception as e:
            logger.error(f"Error alineando rostro en {image_path}: {e}")
            return None

    def extract_embedding(self, aligned_face: np.ndarray) -> Optional[np.ndarray]:
        """
        Extraer embedding de rostro alineado

        Args:
            aligned_face: Imagen de rostro alineada

        Returns:
            Vector embedding o None si hay error
        """
        try:
            faces = self.face_app.get(aligned_face)
            if len(faces) == 0:
                return None

            embedding = faces[0].embedding
            embedding = embedding / np.linalg.norm(embedding)

            return embedding

        except Exception as e:
            logger.error(f"Error extrayendo embedding: {e}")
            return None

    def process_images(self, 
                      images_dir: str, 
                      max_images: Optional[int] = None,
                      save_aligned: bool = True,
                      save_embeddings: bool = True) -> Dict:
        """
        Procesar imágenes de CelebA: alineación y extracción de embeddings

        Args:
            images_dir: Directorio con imágenes de CelebA
            max_images: Máximo número de imágenes a procesar
            save_aligned: Si guardar rostros alineados
            save_embeddings: Si guardar embeddings

        Returns:
            Diccionario con estadísticas del procesamiento
        """
        images_dir = Path(images_dir)

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in images_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]

        if max_images:
            image_files = image_files[:max_images]

        logger.info(f"Procesando {len(image_files)} imágenes")

        identities = self.load_celeba_identities()
        embeddings_by_identity = {}
        processed_count = 0
        failed_count = 0

        for img_file in tqdm(image_files, desc="Procesando imágenes"):
            try:
                aligned_face = self.align_face(str(img_file))
                if aligned_face is None:
                    failed_count += 1
                    continue

                embedding = self.extract_embedding(aligned_face)
                if embedding is None:
                    failed_count += 1
                    continue

                identity_id = identities.get(img_file.name, "unknown")

                if save_aligned:
                    aligned_filename = f"{img_file.stem}_aligned.jpg"
                    aligned_path = self.aligned_dir / aligned_filename
                    cv2.imwrite(str(aligned_path), aligned_face)

                if identity_id not in embeddings_by_identity:
                    embeddings_by_identity[identity_id] = []

                embeddings_by_identity[identity_id].append({
                    'image_name': img_file.name,
                    'embedding': embedding,
                    'aligned_path': str(aligned_path) if save_aligned else None
                })

                processed_count += 1

            except Exception as e:
                logger.error(f"Error procesando {img_file}: {e}")
                failed_count += 1

        if save_embeddings:
            self.save_embeddings(embeddings_by_identity)

        stats = {
            'total_images': len(image_files),
            'processed': processed_count,
            'failed': failed_count,
            'unique_identities': len(embeddings_by_identity),
            'success_rate': processed_count / len(image_files) if image_files else 0
        }

        logger.info(f"Procesamiento completado: {stats}")

        return {
            'embeddings_by_identity': embeddings_by_identity,
            'stats': stats
        }

    def save_embeddings(self, embeddings_by_identity: Dict):
        """Guardar embeddings organizados por identidad"""

        embeddings_file = self.embeddings_dir / "celeba_embeddings_full.pkl"
        with open(embeddings_file, 'wb') as f:
            pickle.dump(embeddings_by_identity, f)

        embeddings_array_file = self.embeddings_dir / "celeba_embeddings_array.npy"
        identity_labels_file = self.embeddings_dir / "celeba_identity_labels.npy"

        all_embeddings = []
        all_labels = []

        for identity_id, identity_data in embeddings_by_identity.items():
            for item in identity_data:
                all_embeddings.append(item['embedding'])
                all_labels.append(identity_id)

        if all_embeddings:
            np.save(embeddings_array_file, np.array(all_embeddings))
            np.save(identity_labels_file, np.array(all_labels))

        logger.info(f"Embeddings guardados en:")
        logger.info(f"  - {embeddings_file}")
        logger.info(f"  - {embeddings_array_file}")
        logger.info(f"  - {identity_labels_file}")

    def load_embeddings(self) -> Optional[Dict]:
        """Cargar embeddings previamente guardados"""
        embeddings_file = self.embeddings_dir / "celeba_embeddings_full.pkl"

        if not embeddings_file.exists():
            logger.warning(f"Archivo de embeddings no encontrado: {embeddings_file}")
            return None

        with open(embeddings_file, 'rb') as f:
            embeddings_by_identity = pickle.load(f)

        logger.info(f"Embeddings cargados: {len(embeddings_by_identity)} identidades")
        return embeddings_by_identity

def main():
    """Función principal para ejecutar el pipeline"""

    processor = CelebAProcessor(
        data_dir="data/celeba",
        model_name="buffalo_l",
        device="cpu"
    )

    images_dir = processor.download_celeba_sample()

    if not any(images_dir.iterdir()):
        logger.info("No se encontraron imágenes en el directorio.")
        logger.info("Por favor, descarga el dataset CelebA y colócalo en:")
        logger.info(f"  {images_dir}")
        return

    results = processor.process_images(
        images_dir=images_dir,
        max_images=100,
        save_aligned=True,
        save_embeddings=True
    )

    stats = results['stats']
    print("\n🎯 Resultados del procesamiento:")
    print(f"  📸 Imágenes procesadas: {stats['processed']}/{stats['total_images']}")
    print(f"  👥 Identidades únicas: {stats['unique_identities']}")
    print(f"  ✅ Tasa de éxito: {stats['success_rate']:.2%}")
    print(f"  ❌ Fallos: {stats['failed']}")

    embeddings_by_identity = results['embeddings_by_identity']
    if embeddings_by_identity:
        print(f"\n📊 Ejemplo de embeddings por identidad:")
        for identity_id, data in list(embeddings_by_identity.items())[:3]:
            print(f"  ID {identity_id}: {len(data)} imágenes")
            if data:
                print(f"    Dimensión embedding: {data[0]['embedding'].shape}")

if __name__ == "__main__":
    main()
