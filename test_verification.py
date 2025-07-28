#!/usr/bin/env python3
"""
Test completo para verificación facial
Prueba comparaciones 1:1, evaluación de umbrales y métricas de rendimiento
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
import itertools
import time

# Importar nuestros módulos
from celeba_pipeline import CelebAProcessor
from embedding_utils import EmbeddingAnalyzer

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VerificationTester:
    """Tester completo para verificación facial"""
    
    def __init__(self, model_name: str = "buffalo_l"):
        """
        Inicializar tester de verificación
        
        Args:
            model_name: Modelo de InsightFace a usar
        """
        self.model_name = model_name
        self.processor = CelebAProcessor(model_name=model_name)
        self.analyzer = EmbeddingAnalyzer()
        
        # Métricas de evaluación
        self.evaluation_metrics = {
            'true_positives': 0,
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'total_comparisons': 0,
            'verification_times': []
        }
        
        # Resultados de comparaciones
        self.comparison_results = []
    
    def extract_embedding_from_path(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extraer embedding de una imagen
        
        Args:
            image_path: Ruta de la imagen
            
        Returns:
            Embedding o None si falla
        """
        try:
            # Alinear rostro
            aligned_face = self.processor.align_face(image_path)
            if aligned_face is None:
                return None
            
            # Extraer embedding
            embedding = self.processor.extract_embedding(aligned_face)
            return embedding
            
        except Exception as e:
            logger.warning(f"Error procesando {image_path}: {e}")
            return None
    
    def test_single_comparison(self, image_path1: str, image_path2: str,
                              ground_truth: bool, threshold: float = 0.4) -> Dict:
        """
        Test de comparación entre dos imágenes
        
        Args:
            image_path1, image_path2: Rutas de las imágenes
            ground_truth: True si son la misma persona, False si no
            threshold: Umbral de decisión
            
        Returns:
            Diccionario con resultados de la comparación
        """
        start_time = time.time()
        
        # Extraer embeddings
        embedding1 = self.extract_embedding_from_path(image_path1)
        embedding2 = self.extract_embedding_from_path(image_path2)
        
        if embedding1 is None or embedding2 is None:
            return {
                'success': False,
                'error': 'No se pudieron extraer embeddings',
                'image1': image_path1,
                'image2': image_path2,
                'ground_truth': ground_truth
            }
        
        # Calcular similitud
        similarity = self.analyzer.calculate_cosine_similarity(embedding1, embedding2)
        euclidean_dist = self.analyzer.calculate_euclidean_distance(embedding1, embedding2)
        
        # Verificar identidad
        verification_result = self.analyzer.verify_identity(
            embedding1, embedding2, threshold=threshold
        )
        
        predicted_match = verification_result['is_match']
        verification_time = time.time() - start_time
        
        # Calcular métricas de confusión
        if ground_truth and predicted_match:
            self.evaluation_metrics['true_positives'] += 1
            classification = 'TP'
        elif ground_truth and not predicted_match:
            self.evaluation_metrics['false_negatives'] += 1
            classification = 'FN'
        elif not ground_truth and predicted_match:
            self.evaluation_metrics['false_positives'] += 1
            classification = 'FP'
        else:  # not ground_truth and not predicted_match
            self.evaluation_metrics['true_negatives'] += 1
            classification = 'TN'
        
        self.evaluation_metrics['total_comparisons'] += 1
        self.evaluation_metrics['verification_times'].append(verification_time)
        
        result = {
            'success': True,
            'image1': image_path1,
            'image2': image_path2,
            'embedding1': embedding1,
            'embedding2': embedding2,
            'cosine_similarity': similarity,
            'euclidean_distance': euclidean_dist,
            'predicted_match': predicted_match,
            'ground_truth': ground_truth,
            'threshold': threshold,
            'verification_time': verification_time,
            'classification': classification,
            'correct_prediction': ground_truth == predicted_match
        }
        
        self.comparison_results.append(result)
        
        return result
    
    def test_positive_pairs(self, image_pairs: List[Tuple[str, str]], 
                           threshold: float = 0.4) -> List[Dict]:
        """
        Test de pares positivos (misma persona)
        
        Args:
            image_pairs: Lista de tuplas con pares de imágenes de la misma persona
            threshold: Umbral de decisión
            
        Returns:
            Lista de resultados
        """
        logger.info(f"🧪 Testing {len(image_pairs)} pares positivos...")
        
        results = []
        for img1, img2 in tqdm(image_pairs, desc="Pares positivos"):
            result = self.test_single_comparison(img1, img2, ground_truth=True, threshold=threshold)
            results.append(result)
        
        return results
    
    def test_negative_pairs(self, image_pairs: List[Tuple[str, str]], 
                           threshold: float = 0.4) -> List[Dict]:
        """
        Test de pares negativos (personas diferentes)
        
        Args:
            image_pairs: Lista de tuplas con pares de imágenes de personas diferentes
            threshold: Umbral de decisión
            
        Returns:
            Lista de resultados
        """
        logger.info(f"🧪 Testing {len(image_pairs)} pares negativos...")
        
        results = []
        for img1, img2 in tqdm(image_pairs, desc="Pares negativos"):
            result = self.test_single_comparison(img1, img2, ground_truth=False, threshold=threshold)
            results.append(result)
        
        return results
    
    def create_test_pairs_from_celeba(self, max_positive_pairs: int = 50, 
                                     max_negative_pairs: int = 50) -> Tuple[List, List]:
        """
        Crear pares de test usando dataset CelebA
        
        Args:
            max_positive_pairs: Máximo número de pares positivos
            max_negative_pairs: Máximo número de pares negativos
            
        Returns:
            Tupla con (pares_positivos, pares_negativos)
        """
        celeba_dir = Path("data/celeba/img_align_celeba")
        identity_file = Path("data/celeba/identity_CelebA.txt")
        
        if not celeba_dir.exists() or not identity_file.exists():
            logger.warning("⚠️  Dataset CelebA no completo")
            return [], []
        
        # Cargar mapeo de identidades
        identities = {}
        with open(identity_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    image_name = parts[0]
                    identity_id = int(parts[1])
                    identities[image_name] = identity_id
        
        # Agrupar imágenes por identidad
        identity_groups = {}
        for image_name, identity_id in identities.items():
            image_path = celeba_dir / image_name
            if image_path.exists():
                if identity_id not in identity_groups:
                    identity_groups[identity_id] = []
                identity_groups[identity_id].append(str(image_path))
        
        # Crear pares positivos (misma identidad)
        positive_pairs = []
        for identity_id, images in identity_groups.items():
            if len(images) >= 2:
                # Tomar pares de la misma identidad
                for i in range(min(2, len(images))):
                    for j in range(i + 1, min(3, len(images))):
                        positive_pairs.append((images[i], images[j]))
                        if len(positive_pairs) >= max_positive_pairs:
                            break
                if len(positive_pairs) >= max_positive_pairs:
                    break
        
        # Crear pares negativos (identidades diferentes)
        negative_pairs = []
        identity_list = list(identity_groups.keys())
        
        for i in range(min(max_negative_pairs, len(identity_list))):
            for j in range(i + 1, len(identity_list)):
                if len(identity_groups[identity_list[i]]) > 0 and len(identity_groups[identity_list[j]]) > 0:
                    img1 = identity_groups[identity_list[i]][0]
                    img2 = identity_groups[identity_list[j]][0]
                    negative_pairs.append((img1, img2))
                    
                    if len(negative_pairs) >= max_negative_pairs:
                        break
            if len(negative_pairs) >= max_negative_pairs:
                break
        
        logger.info(f"✅ Creados {len(positive_pairs)} pares positivos y {len(negative_pairs)} pares negativos")
        
        return positive_pairs, negative_pairs
    
    def create_test_pairs_from_samples(self) -> Tuple[List, List]:
        """
        Crear pares de test usando imágenes de muestra
        
        Returns:
            Tupla con (pares_positivos, pares_negativos)
        """
        sample_dir = Path("data/sample_images")
        
        if not sample_dir.exists():
            return [], []
        
        image_files = list(sample_dir.glob("*.jpg"))
        
        if len(image_files) < 2:
            return [], []
        
        # Para imágenes de muestra, asumir que las primeras dos son la misma persona
        # y el resto son personas diferentes
        positive_pairs = []
        negative_pairs = []
        
        if len(image_files) >= 2:
            # Pares positivos: comparar primeras 2 imágenes consigo mismas
            positive_pairs.append((str(image_files[0]), str(image_files[0])))
            if len(image_files) > 1:
                positive_pairs.append((str(image_files[1]), str(image_files[1])))
        
        # Pares negativos: comparar imágenes diferentes
        for i in range(len(image_files)):
            for j in range(i + 1, len(image_files)):
                negative_pairs.append((str(image_files[i]), str(image_files[j])))
                if len(negative_pairs) >= 10:  # Limitar número
                    break
            if len(negative_pairs) >= 10:
                break
        
        return positive_pairs, negative_pairs
    
    def evaluate_threshold_range(self, thresholds: List[float]) -> Dict:
        """
        Evaluar rendimiento para diferentes umbrales
        
        Args:
            thresholds: Lista de umbrales a evaluar
            
        Returns:
            Diccionario con métricas por umbral
        """
        if not self.comparison_results:
            logger.warning("⚠️  No hay resultados de comparación para evaluar")
            return {}
        
        logger.info(f"📊 Evaluando {len(thresholds)} umbrales...")
        
        threshold_metrics = {}
        
        for threshold in thresholds:
            tp = tn = fp = fn = 0
            
            for result in self.comparison_results:
                if not result['success']:
                    continue
                
                # Recalcular decisión con nuevo umbral
                predicted_match = result['cosine_similarity'] >= threshold
                ground_truth = result['ground_truth']
                
                if ground_truth and predicted_match:
                    tp += 1
                elif ground_truth and not predicted_match:
                    fn += 1
                elif not ground_truth and predicted_match:
                    fp += 1
                else:
                    tn += 1
            
            # Calcular métricas
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            threshold_metrics[threshold] = {
                'threshold': threshold,
                'true_positives': tp,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'accuracy': accuracy
            }
        
        return threshold_metrics
    
    def calculate_metrics(self) -> Dict:
        """Calcular métricas de evaluación finales"""
        metrics = self.evaluation_metrics
        
        total = metrics['total_comparisons']
        if total == 0:
            return {}
        
        tp = metrics['true_positives']
        tn = metrics['true_negatives']
        fp = metrics['false_positives']
        fn = metrics['false_negatives']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / total
        
        avg_time = np.mean(metrics['verification_times']) if metrics['verification_times'] else 0
        
        return {
            'total_comparisons': total,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'avg_verification_time': avg_time
        }
    
    def plot_roc_curve(self, save_path: str = "roc_curve.png"):
        """Generar curva ROC"""
        if not self.comparison_results:
            logger.warning("⚠️  No hay resultados para generar ROC")
            return
        
        # Extraer similitudes y etiquetas verdaderas
        similarities = []
        true_labels = []
        
        for result in self.comparison_results:
            if result['success']:
                similarities.append(result['cosine_similarity'])
                true_labels.append(1 if result['ground_truth'] else 0)
        
        if len(similarities) < 2:
            logger.warning("⚠️  Insuficientes datos para ROC")
            return
        
        # Calcular ROC
        fpr, tpr, thresholds = roc_curve(true_labels, similarities)
        roc_auc = auc(fpr, tpr)
        
        # Plotear
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Verificación Facial ({self.model_name})')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"📈 Curva ROC guardada: {save_path}")
        plt.show()
        
        return roc_auc
    
    def plot_threshold_analysis(self, threshold_metrics: Dict, save_path: str = "threshold_analysis.png"):
        """Visualizar análisis de umbrales"""
        if not threshold_metrics:
            return
        
        thresholds = list(threshold_metrics.keys())
        precisions = [threshold_metrics[t]['precision'] for t in thresholds]
        recalls = [threshold_metrics[t]['recall'] for t in thresholds]
        f1_scores = [threshold_metrics[t]['f1_score'] for t in thresholds]
        accuracies = [threshold_metrics[t]['accuracy'] for t in thresholds]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(thresholds, precisions, 'o-', label='Precision', color='blue')
        plt.xlabel('Threshold')
        plt.ylabel('Precision')
        plt.title('Precision vs Threshold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(thresholds, recalls, 'o-', label='Recall', color='green')
        plt.xlabel('Threshold')
        plt.ylabel('Recall')
        plt.title('Recall vs Threshold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(thresholds, f1_scores, 'o-', label='F1-Score', color='red')
        plt.xlabel('Threshold')
        plt.ylabel('F1-Score')
        plt.title('F1-Score vs Threshold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(thresholds, accuracies, 'o-', label='Accuracy', color='purple')
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Threshold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Análisis de umbrales guardado: {save_path}")
        plt.show()
    
    def plot_confusion_matrix(self, threshold: float = 0.4, save_path: str = "confusion_matrix.png"):
        """Generar matriz de confusión"""
        if not self.comparison_results:
            return
        
        y_true = []
        y_pred = []
        
        for result in self.comparison_results:
            if result['success']:
                y_true.append(1 if result['ground_truth'] else 0)
                predicted_match = result['cosine_similarity'] >= threshold
                y_pred.append(1 if predicted_match else 0)
        
        if not y_true:
            return
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Different Person', 'Same Person'],
                   yticklabels=['Different Person', 'Same Person'])
        plt.title(f'Confusion Matrix (Threshold = {threshold})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"🔄 Matriz de confusión guardada: {save_path}")
        plt.show()
    
    def print_evaluation_summary(self):
        """Mostrar resumen de evaluación"""
        metrics = self.calculate_metrics()
        
        if not metrics:
            print("⚠️  No hay métricas para mostrar")
            return
        
        print(f"\n{'='*60}")
        print(f"📊 RESUMEN DE VERIFICACIÓN FACIAL - {self.model_name.upper()}")
        print(f"{'='*60}")
        print(f"📈 Total de comparaciones: {metrics['total_comparisons']}")
        print(f"✅ Verdaderos positivos: {metrics['true_positives']}")
        print(f"✅ Verdaderos negativos: {metrics['true_negatives']}")
        print(f"❌ Falsos positivos: {metrics['false_positives']}")
        print(f"❌ Falsos negativos: {metrics['false_negatives']}")
        print(f"")
        print(f"🎯 Precision: {metrics['precision']:.3f}")
        print(f"🎯 Recall: {metrics['recall']:.3f}")
        print(f"🎯 F1-Score: {metrics['f1_score']:.3f}")
        print(f"🎯 Accuracy: {metrics['accuracy']:.3f}")
        print(f"⏱️  Tiempo promedio: {metrics['avg_verification_time']:.3f}s")
        print(f"{'='*60}")

def test_verification_with_celeba():
    """Test de verificación con dataset CelebA"""
    logger.info("🧪 Test de verificación con CelebA")
    
    tester = VerificationTester(model_name="buffalo_l")
    
    # Crear pares de test
    positive_pairs, negative_pairs = tester.create_test_pairs_from_celeba(
        max_positive_pairs=30,
        max_negative_pairs=30
    )
    
    if not positive_pairs and not negative_pairs:
        logger.warning("⚠️  No se pudieron crear pares de test con CelebA")
        return
    
    # Testear con umbral por defecto
    threshold = 0.4
    
    # Test pares positivos
    positive_results = tester.test_positive_pairs(positive_pairs, threshold)
    
    # Test pares negativos  
    negative_results = tester.test_negative_pairs(negative_pairs, threshold)
    
    # Mostrar resultados
    tester.print_evaluation_summary()
    
    # Evaluación de umbrales
    thresholds = np.arange(0.2, 0.8, 0.05)
    threshold_metrics = tester.evaluate_threshold_range(thresholds)
    
    # Visualizaciones
    tester.plot_roc_curve("roc_curve_celeba.png")
    tester.plot_threshold_analysis(threshold_metrics, "threshold_analysis_celeba.png")
    tester.plot_confusion_matrix(threshold, "confusion_matrix_celeba.png")
    
    return tester

def test_verification_with_samples():
    """Test de verificación con imágenes de muestra"""
    logger.info("🧪 Test de verificación con imágenes de muestra")
    
    tester = VerificationTester(model_name="buffalo_l")
    
    # Crear pares de test
    positive_pairs, negative_pairs = tester.create_test_pairs_from_samples()
    
    if not positive_pairs and not negative_pairs:
        logger.warning("⚠️  No se pudieron crear pares de test con imágenes de muestra")
        return
    
    threshold = 0.4
    
    # Test pares
    if positive_pairs:
        positive_results = tester.test_positive_pairs(positive_pairs, threshold)
    
    if negative_pairs:
        negative_results = tester.test_negative_pairs(negative_pairs, threshold)
    
    # Mostrar resultados
    tester.print_evaluation_summary()
    
    return tester

def demo_single_verification():
    """Demo de verificación individual con visualización"""
    logger.info("🧪 Demo de verificación individual")
    
    from demo_pipeline import FaceVerificationDemo
    
    # Crear demo
    demo = FaceVerificationDemo(model_name="buffalo_l")
    
    # Crear imágenes de muestra si no existen
    sample_dir = demo.create_sample_images()
    image_files = list(sample_dir.glob("*.jpg"))
    
    if len(image_files) >= 2:
        # Demo de verificación
        demo.demo_face_verification(str(image_files[0]), str(image_files[1]))
        demo.demo_face_verification(str(image_files[0]), str(image_files[0]))  # Misma imagen
    else:
        logger.warning("⚠️  No hay suficientes imágenes para el demo")

def main():
    """Función principal de testing de verificación"""
    print("🔍 TESTING DE VERIFICACIÓN FACIAL")
    print("="*60)
    
    try:
        # Test 1: Demo individual
        demo_single_verification()
        
        # Test 2: Verificación con imágenes de muestra
        test_verification_with_samples()
        
        # Test 3: Verificación con CelebA (si está disponible)
        test_verification_with_celeba()
        
        print(f"\n🎉 Testing de verificación completado!")
        print(f"📁 Revisa los archivos *.png para ver las visualizaciones")
        
    except Exception as e:
        logger.error(f"❌ Error en testing de verificación: {e}")
        raise

if __name__ == "__main__":
    main()