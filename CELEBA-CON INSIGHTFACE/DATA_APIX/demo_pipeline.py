#!/usr/bin/env python3
"""
Demostración completa del pipeline de verificación facial con CelebA e InsightFace
Incluye ejemplos de uso para alineación, extracción de embeddings y verificación
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from celeba_pipeline import CelebAProcessor
from embedding_utils import EmbeddingAnalyzer
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceVerificationDemo:
    """Demo del sistema de verificación facial"""
    
    def __init__(self, model_name: str = "buffalo_l"):
        self.processor = CelebAProcessor(model_name=model_name)
        self.analyzer = None
    
    def create_sample_images(self):
        """Crear imágenes de muestra para testing si no hay CelebA disponible"""
        sample_dir = Path("data/sample_images")
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("📷 Creando imágenes de muestra para demo...")
        
        for i in range(5):
            img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            cv2.circle(img, (56, 56), 40, (200, 180, 160), -1)
            cv2.circle(img, (45, 45), 5, (0, 0, 0), -1)
            cv2.circle(img, (67, 45), 5, (0, 0, 0), -1)
            cv2.circle(img, (56, 56), 3, (100, 50, 50), -1)
            cv2.ellipse(img, (56, 70), (10, 5), 0, 0, 180, (50, 50, 50), 2)
            
            if i < 2:      # Identidad 1
                cv2.circle(img, (56, 35), 15, (150, 100, 50), -1)
            elif i < 4:    # Identidad 2
                cv2.rectangle(img, (45, 30), (67, 40), (200, 150, 100), -1)
            
            sample_path = sample_dir / f"sample_{i:03d}.jpg"
            cv2.imwrite(str(sample_path), img)
        
        logger.info(f"✅ Imágenes de muestra creadas en: {sample_dir}")
        return sample_dir
    
    def demo_face_alignment(self, image_path: str):
        logger.info(f"🔧 Demostrando alineación facial para: {image_path}")
        
        original_img = cv2.imread(image_path)
        if original_img is None:
            logger.error(f"No se pudo cargar la imagen: {image_path}")
            return None
        
        aligned_face = self.processor.align_face(image_path)
        if aligned_face is None:
            logger.warning(f"No se detectó rostro en: {image_path}")
            return None
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.title("Imagen Original")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB))
        plt.title("Rostro Alineado")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.text(0.1, 0.8, f"Imagen Original:", fontsize=12, weight='bold')
        plt.text(0.1, 0.7, f"  Tamaño: {original_img.shape[:2]}", fontsize=10)
        plt.text(0.1, 0.5, f"Rostro Alineado:", fontsize=12, weight='bold')
        plt.text(0.1, 0.4, f"  Tamaño: {aligned_face.shape[:2]}", fontsize=10)
        plt.text(0.1, 0.3, f"  Formato: Normalizado", fontsize=10)
        plt.xlim(0, 1); plt.ylim(0, 1); plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"alignment_demo_{Path(image_path).stem}.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return aligned_face
    
    def demo_embedding_extraction(self, image_path: str):
        logger.info(f"🧠 Demostrando extracción de embeddings para: {image_path}")
        
        aligned_face = self.processor.align_face(image_path)
        if aligned_face is None:
            logger.error(f"No se pudo alinear rostro en: {image_path}")
            return None
        
        embedding = self.processor.extract_embedding(aligned_face)
        if embedding is None:
            logger.error(f"No se pudo extraer embedding de: {image_path}")
            return None
        
        print(f"\n📊 Información del Embedding:")
        print(f"  📁 Archivo: {Path(image_path).name}")
        print(f"  📏 Dimensiones: {embedding.shape}")
        print(f"  📈 Rango: [{embedding.min():.3f}, {embedding.max():.3f}]")
        print(f"  📉 Media: {embedding.mean():.3f}")
        print(f"  📊 Desviación estándar: {embedding.std():.3f}")
        print(f"  🔢 Norma L2: {np.linalg.norm(embedding):.3f}")
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB))
        plt.title("Rostro Alineado"); plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.hist(embedding, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.title("Distribución del Embedding"); plt.xlabel("Valor"); plt.ylabel("Frecuencia")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        dim = int(np.sqrt(len(embedding)))
        if dim * dim == len(embedding):
            embedding_2d = embedding.reshape(dim, dim)
        else:
            size = int(np.ceil(np.sqrt(len(embedding))))
            padded = np.zeros(size * size)
            padded[:len(embedding)] = embedding
            embedding_2d = padded.reshape(size, size)
        plt.imshow(embedding_2d, cmap='viridis'); plt.title("Embedding como Imagen")
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(f"embedding_demo_{Path(image_path).stem}.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return embedding
    
    def demo_face_verification(self, image_path1: str, image_path2: str):
        logger.info(f"🔍 Demostrando verificación facial:")
        logger.info(f"  Imagen 1: {image_path1}")
        logger.info(f"  Imagen 2: {image_path2}")
        
        embedding1 = self.demo_embedding_extraction(image_path1)
        embedding2 = self.demo_embedding_extraction(image_path2)
        if embedding1 is None or embedding2 is None:
            logger.error("No se pudieron extraer embeddings de ambas imágenes")
            return
        
        if self.analyzer is None:
            self.analyzer = EmbeddingAnalyzer()
        
        cosine_sim = self.analyzer.calculate_cosine_similarity(embedding1, embedding2)
        euclidean_dist = self.analyzer.calculate_euclidean_distance(embedding1, embedding2)
        
        thresholds = [0.3, 0.4, 0.5, 0.6]
        results = []
        for threshold in thresholds:
            result = self.analyzer.verify_identity(embedding1, embedding2, threshold=threshold)
            results.append(result)
        
        print(f"\n🎯 Resultados de Verificación:")
        print(f"  📐 Similitud Coseno: {cosine_sim:.4f}")
        print(f"  📏 Distancia Euclidiana: {euclidean_dist:.4f}")
        print(f"\n  🎚️  Verificación por Umbral:")
        for threshold, result in zip(thresholds, results):
            match_text = "✅ MATCH" if result['is_match'] else "❌ NO MATCH"
            print(f"    Umbral {threshold}: {match_text} (similitud: {result['score']:.4f})")
        
        self.visualize_verification_result(image_path1, image_path2, cosine_sim, euclidean_dist, results)
    
    def visualize_verification_result(self, image_path1: str, image_path2: str, 
                                      cosine_sim: float, euclidean_dist: float, results: list):
        aligned1 = self.processor.align_face(image_path1)
        aligned2 = self.processor.align_face(image_path2)
        
        plt.figure(figsize=(15, 8))
        plt.subplot(2, 4, 1)
        if aligned1 is not None:
            plt.imshow(cv2.cvtColor(aligned1, cv2.COLOR_BGR2RGB))
        plt.title(f"Imagen 1\n{Path(image_path1).name}")
        plt.axis('off')
        
        plt.subplot(2, 4, 2)
        if aligned2 is not None:
            plt.imshow(cv2.cvtColor(aligned2, cv2.COLOR_BGR2RGB))
        plt.title(f"Imagen 2\n{Path(image_path2).name}")
        plt.axis('off')
        
        plt.subplot(2, 4, 3)
        metrics = ['Coseno', 'Euclidiana']
        values = [cosine_sim, euclidean_dist]
        colors = ['blue', 'red']
        bars = plt.bar(metrics, values, color=colors, alpha=0.7)
        plt.title("Métricas de Similitud"); plt.ylabel("Valor")
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{value:.3f}', ha='center', va='bottom')
        
        plt.subplot(2, 4, 4)
        thresholds = [r['threshold'] for r in results]
        matches = [1 if r['is_match'] else 0 for r in results]
        colors = ['green' if m else 'red' for m in matches]
        bars = plt.bar(range(len(thresholds)), matches, color=colors, alpha=0.7)
        plt.xticks(range(len(thresholds)), [f'{t:.1f}' for t in thresholds])
        plt.title("Decisión por Umbral"); plt.xlabel("Umbral"); plt.ylabel("Match (1) / No Match (0)")
        plt.ylim(0, 1.2)
        
        plt.subplot(2, 1, 2)
        info_text = f"""
📊 RESULTADOS DE VERIFICACIÓN FACIAL

🔍 Métricas de Similitud:
   • Similitud Coseno: {cosine_sim:.4f}
   • Distancia Euclidiana: {euclidean_dist:.4f}

🎯 Decisiones por Umbral:
"""
        for result in results:
            match_symbol = "✅" if result['is_match'] else "❌"
            info_text += f"   • Umbral {result['threshold']:.1f}: {match_symbol} ({result['score']:.4f})\n"
        
        info_text += f"""
💡 Interpretación:
   • Similitud Coseno más alta = Mayor similitud
   • Distancia Euclidiana más baja = Mayor similitud
   • Umbral óptimo típico: 0.4-0.6 para similitud coseno
"""
        plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, 
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"verification_result_{Path(image_path1).stem}_vs_{Path(image_path2).stem}.png", 
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_demo(self):
        logger.info("🚀 Iniciando demostración completa del pipeline")
        
        sample_dir = self.create_sample_images()
        image_files = list(sample_dir.glob("*.jpg"))
        
        if len(image_files) < 2:
            logger.error("Se necesitan al menos 2 imágenes para la demo")
            return
        
        print(f"\n📁 Imágenes encontradas: {len(image_files)}")
        for img in image_files:
            print(f"  • {img.name}")
        
        print(f"\n{'='*60}")
        print("🔧 DEMO: ALINEACIÓN FACIAL")
        print(f"{'='*60}")
        self.demo_face_alignment(str(image_files[0]))
        
        print(f"\n{'='*60}")
        print("🧠 DEMO: EXTRACCIÓN DE EMBEDDINGS")
        print(f"{'='*60}")
        self.demo_embedding_extraction(str(image_files[0]))
        
        print(f"\n{'='*60}")
        print("🔍 DEMO: VERIFICACIÓN FACIAL")
        print(f"{'='*60}")
        print("\n🔄 Comparación 1: Imagen consigo misma (debería ser MATCH)")
        self.demo_face_verification(str(image_files[0]), str(image_files[0]))
        
        if len(image_files) > 1:
            print(f"\n🆚 Comparación 2: Dos imágenes diferentes")
            self.demo_face_verification(str(image_files[0]), str(image_files[1]))
        
        print(f"\n{'='*60}")
        print("✅ DEMOSTRACIÓN COMPLETADA")
        print(f"{'='*60}")
        print("📁 Archivos generados:")
        print("  • alignment_demo_*.png - Demostración de alineación")
        print("  • embedding_demo_*.png - Visualización de embeddings")
        print("  • verification_result_*.png - Resultados de verificación")

def main():
    print("🎭 DEMO: Pipeline de Verificación Facial con InsightFace")
    print("="*60)
    demo = FaceVerificationDemo(model_name="buffalo_l")
    demo.run_complete_demo()

if __name__ == "__main__":
    main()
