#!/usr/bin/env python3
"""
Test completo para extracción de embeddings faciales
Prueba el pipeline con diferentes imágenes y modelos de InsightFace
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from tqdm import tqdm

# Importar nuestros módulos
from celeba_pipeline import CelebAProcessor
from embedding_utils import EmbeddingAnalyzer

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingTester:
    """Tester completo para extracción de embeddings"""
    
    def __init__(self, model_name: str = "buffalo_l"):
        self.model_name = model_name
        self.processor = None
        self.analyzer = EmbeddingAnalyzer()
        self.performance_metrics = {
            'total_images': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'avg_extraction_time': 0.0,
            'times': []
        }
        self.load_processor()
    
    def load_processor(self):
        try:
            logger.info(f"Inicializando procesador con modelo: {self.model_name}")
            self.processor = CelebAProcessor(model_name=self.model_name)
            logger.info("✅ Procesador inicializado correctamente")
        except Exception as e:
            logger.error(f"❌ Error inicializando procesador: {e}")
            raise
    
    def test_single_image(self, image_path: str, show_results: bool = True) -> Dict:
        logger.info(f"🧪 Testing extracción de embedding: {image_path}")
        start_time = time.time()
        
        try:
            original_img = cv2.imread(image_path)
            if original_img is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path}")
            
            align_start = time.time()
            aligned_face = self.processor.align_face(image_path)
            align_time = time.time() - align_start
            
            if aligned_face is None:
                raise ValueError("No se detectó rostro en la imagen")
            
            embed_start = time.time()
            embedding = self.processor.extract_embedding(aligned_face)
            embed_time = time.time() - embed_start
            
            if embedding is None:
                raise ValueError("No se pudo extraer embedding")
            
            total_time = time.time() - start_time
            metrics = self.analyze_embedding(embedding)
            
            if show_results:
                self.visualize_extraction_result(
                    original_img, aligned_face, embedding, 
                    image_path, metrics, total_time
                )
            
            result = {
                'success': True,
                'image_path': image_path,
                'embedding': embedding,
                'metrics': metrics,
                'times': {
                    'total': total_time,
                    'alignment': align_time,
                    'extraction': embed_time
                },
                'image_shapes': {
                    'original': original_img.shape,
                    'aligned': aligned_face.shape
                },
                'error': None
            }
            
            self.performance_metrics['successful_extractions'] += 1
            self.performance_metrics['times'].append(total_time)
            logger.info(f"✅ Extracción exitosa en {total_time:.3f}s")
            return result
            
        except Exception as e:
            error_msg = f"Error procesando {image_path}: {e}"
            logger.error(f"❌ {error_msg}")
            self.performance_metrics['failed_extractions'] += 1
            return {
                'success': False,
                'image_path': image_path,
                'embedding': None,
                'metrics': None,
                'times': None,
                'image_shapes': None,
                'error': str(e)
            }
        finally:
            self.performance_metrics['total_images'] += 1
    
    def analyze_embedding(self, embedding: np.ndarray) -> Dict:
        return {
            'dimension': embedding.shape[0],
            'mean': float(np.mean(embedding)),
            'std': float(np.std(embedding)),
            'min': float(np.min(embedding)),
            'max': float(np.max(embedding)),
            'l2_norm': float(np.linalg.norm(embedding)),
            'l1_norm': float(np.linalg.norm(embedding, ord=1)),
            'zero_ratio': float(np.sum(embedding == 0) / len(embedding)),
            'range': float(np.max(embedding) - np.min(embedding))
        }
    
    def test_multiple_images(self, image_paths: List[str], max_images: Optional[int] = None) -> List[Dict]:
        if max_images:
            image_paths = image_paths[:max_images]
        
        logger.info(f"🧪 Testing {len(image_paths)} imágenes...")
        results = []
        
        for image_path in tqdm(image_paths, desc="Procesando imágenes"):
            result = self.test_single_image(image_path, show_results=False)
            results.append(result)
        
        self.calculate_performance_stats()
        return results
    
    def test_directory(self, images_dir: str, max_images: Optional[int] = None) -> List[Dict]:
        images_dir = Path(images_dir)
        if not images_dir.exists():
            raise ValueError(f"Directorio no encontrado: {images_dir}")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = [
            str(img_file) for img_file in images_dir.rglob("*")
            if img_file.suffix.lower() in image_extensions
        ]
        
        if not image_paths:
            raise ValueError(f"No se encontraron imágenes en: {images_dir}")
        
        logger.info(f"📁 Encontradas {len(image_paths)} imágenes en {images_dir}")
        return self.test_multiple_images(image_paths, max_images)
    
    def benchmark_models(self, image_paths: List[str], models: List[str] = None) -> Dict:
        if models is None:
            models = ['buffalo_l', 'arcface_r50']
        
        logger.info(f"🏁 Benchmark de modelos: {models}")
        benchmark_results = {}
        
        for model_name in models:
            logger.info(f"📊 Testing modelo: {model_name}")
            self.model_name = model_name
            self.performance_metrics = {
                'total_images': 0,
                'successful_extractions': 0,
                'failed_extractions': 0,
                'avg_extraction_time': 0.0,
                'times': []
            }
            
            try:
                self.load_processor()
                sample_images = image_paths[:min(10, len(image_paths))]
                results = self.test_multiple_images(sample_images)
                
                benchmark_results[model_name] = {
                    'performance_metrics': self.performance_metrics.copy(),
                    'sample_results': results,
                    'success_rate': self.performance_metrics['successful_extractions'] / 
                                   max(1, self.performance_metrics['total_images']),
                    'avg_time': self.performance_metrics['avg_extraction_time']
                }
                
                logger.info(f"✅ Modelo {model_name}: "
                          f"{self.performance_metrics['successful_extractions']}/"
                          f"{self.performance_metrics['total_images']} exitosas")
            except Exception as e:
                logger.error(f"❌ Error con modelo {model_name}: {e}")
                benchmark_results[model_name] = {
                    'error': str(e),
                    'performance_metrics': None,
                    'sample_results': None,
                    'success_rate': 0.0,
                    'avg_time': 0.0
                }
        
        return benchmark_results
    
    def calculate_performance_stats(self):
        times = self.performance_metrics['times']
        
        if times:
            self.performance_metrics['avg_extraction_time'] = np.mean(times)
            self.performance_metrics['std_extraction_time'] = np.std(times)
            self.performance_metrics['min_extraction_time'] = np.min(times)
            self.performance_metrics['max_extraction_time'] = np.max(times)
        
        success_rate = (self.performance_metrics['successful_extractions'] / 
                       max(1, self.performance_metrics['total_images']))
        self.performance_metrics['success_rate'] = success_rate
    
    def visualize_extraction_result(self, original_img: np.ndarray, aligned_face: np.ndarray,
                                   embedding: np.ndarray, image_path: str,
                                   metrics: Dict, total_time: float):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0, 0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f"Original\n{Path(image_path).name}")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title(f"Alineado\n{aligned_face.shape[:2]}")
        axes[0, 1].axis('off')
        
        embedding_dim = int(np.sqrt(len(embedding)))
        if embedding_dim * embedding_dim == len(embedding):
            embedding_2d = embedding.reshape(embedding_dim, embedding_dim)
        else:
            size = int(np.ceil(np.sqrt(len(embedding))))
            padded = np.zeros(size * size)
            padded[:len(embedding)] = embedding
            embedding_2d = padded.reshape(size, size)
        
        im = axes[0, 2].imshow(embedding_2d, cmap='viridis')
        axes[0, 2].set_title(f"Embedding Visualizado\n{embedding.shape}")
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2])
        
        axes[1, 0].hist(embedding, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 0].set_title("Distribución del Embedding")
        axes[1, 0].set_xlabel("Valor")
        axes[1, 0].set_ylabel("Frecuencia")
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].axis('off')
        metrics_text = f"""
📊 MÉTRICAS DEL EMBEDDING

🔢 Dimensiones: {metrics['dimension']}
📈 Media: {metrics['mean']:.4f}
📊 Desv. Est.: {metrics['std']:.4f}
📉 Mín: {metrics['min']:.4f}
📈 Máx: {metrics['max']:.4f}
🔢 Norma L2: {metrics['l2_norm']:.4f}
🔢 Norma L1: {metrics['l1_norm']:.4f}
⚫ Ceros: {metrics['zero_ratio']:.2%}
📏 Rango: {metrics['range']:.4f}

⏱️ Tiempo total: {total_time:.3f}s
🤖 Modelo: {self.model_name}
"""
        axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        axes[1, 2].plot(embedding[:50], 'o-', alpha=0.7, markersize=3)
        axes[1, 2].set_title("Primeras 50 Componentes")
        axes[1, 2].set_xlabel("Índice")
        axes[1, 2].set_ylabel("Valor")
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = f"test_embedding_{Path(image_path).stem}_{self.model_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Visualización guardada: {output_path}")
        plt.show()
    
    def print_performance_summary(self):
        metrics = self.performance_metrics
        print(f"\n{'='*60}")
        print(f"📊 RESUMEN DE RENDIMIENTO - {self.model_name.upper()}")
        print(f"{'='*60}")
        print(f"📸 Total de imágenes: {metrics['total_images']}")
        print(f"✅ Extracciones exitosas: {metrics['successful_extractions']}")
        print(f"❌ Extracciones fallidas: {metrics['failed_extractions']}")
        print(f"📈 Tasa de éxito: {metrics.get('success_rate', 0):.2%}")
        if metrics['times']:
            print(f"⏱️  Tiempo promedio: {metrics['avg_extraction_time']:.3f}s")
            print(f"⏱️  Tiempo mínimo: {metrics.get('min_extraction_time', 0):.3f}s")
            print(f"⏱️  Tiempo máximo: {metrics.get('max_extraction_time', 0):.3f}s")
            print(f"⏱️  Desv. estándar: {metrics.get('std_extraction_time', 0):.3f}s")
        print(f"{'='*60}")

def test_sample_images():
    logger.info("🧪 Test 1: Imágenes de muestra")
    tester = EmbeddingTester(model_name="buffalo_l")
    sample_dir = Path("data/sample_images")
    
    if sample_dir.exists():
        results = tester.test_directory(str(sample_dir), max_images=5)
        tester.print_performance_summary()
        return results
    else:
        logger.warning("⚠️  No se encontraron imágenes de muestra")
        return []

def test_celeba_sample():
    logger.info("🧪 Test 2: Muestra de CelebA")
    celeba_dir = Path("data/celeba/img_align_celeba")
    
    if not celeba_dir.exists():
        logger.warning("⚠️  Dataset CelebA no encontrado")
        logger.info("💡 Ejecuta primero: python download_celeba.py")
        return []
    
    tester = EmbeddingTester(model_name="buffalo_l")
    results = tester.test_directory(str(celeba_dir), max_images=20)
    tester.print_performance_summary()
    return results

def test_model_comparison():
    logger.info("🧪 Test 3: Comparación de modelos")
    test_images = []
    
    sample_dir = Path("data/sample_images")
    if sample_dir.exists():
        test_images.extend([str(f) for f in sample_dir.glob("*.jpg")][:5])
    
    celeba_dir = Path("data/celeba/img_align_celeba")
    if celeba_dir.exists():
        test_images.extend([str(f) for f in celeba_dir.glob("*.jpg")][:5])
    
    if not test_images:
        logger.warning("⚠️  No se encontraron imágenes para el test")
        return
    
    tester = EmbeddingTester()
    benchmark_results = tester.benchmark_models(
        test_images, 
        models=['buffalo_l', 'arcface_r50']
    )
    
    print(f"\n{'='*80}")
    print("🏁 BENCHMARK DE MODELOS")
    print(f"{'='*80}")
    
    for model, results in benchmark_results.items():
        if results.get('error'):
            print(f"❌ {model}: Error - {results['error']}")
        else:
            metrics = results['performance_metrics']
            print(f"🤖 {model}:")
            print(f"   ✅ Tasa de éxito: {results['success_rate']:.2%}")
            print(f"   ⏱️  Tiempo promedio: {results['avg_time']:.3f}s")
            print(f"   📊 Procesadas: {metrics['successful_extractions']}/{metrics['total_images']}")
    
    print(f"{'='*80}")

def main():
    print("🧪 TESTING DE EXTRACCIÓN DE EMBEDDINGS")
    print("="*60)
    
    try:
        test_sample_images()
        test_celeba_sample()
        test_model_comparison()
        print(f"\n🎉 Testing completado exitosamente!")
        print(f"📁 Revisa los archivos test_embedding_*.png para ver los resultados")
    except Exception as e:
        logger.error(f"❌ Error en testing: {e}")
        raise

if __name__ == "__main__":
    main()
