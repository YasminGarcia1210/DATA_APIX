#!/usr/bin/env python3
"""
Utilidades para análisis y verificación de embeddings faciales
Incluye cálculo de similitudes, métricas de evaluación y visualización
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class EmbeddingAnalyzer:
    """Analizador de embeddings faciales"""
    
    def __init__(self, embeddings_file: Optional[str] = None):
        """
        Inicializar analizador
        
        Args:
            embeddings_file: Ruta al archivo de embeddings (.pkl)
        """
        self.embeddings_by_identity = {}
        self.embeddings_matrix = None
        self.identity_labels = []
        
        if embeddings_file:
            self.load_embeddings(embeddings_file)
    
    def load_embeddings(self, embeddings_file: str):
        """Cargar embeddings desde archivo"""
        embeddings_file = Path(embeddings_file)
        
        if not embeddings_file.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {embeddings_file}")
        
        with open(embeddings_file, 'rb') as f:
            self.embeddings_by_identity = pickle.load(f)
        
        # Crear matriz de embeddings y etiquetas
        self._create_embeddings_matrix()
        logger.info(f"Embeddings cargados: {len(self.embeddings_by_identity)} identidades")
    
    def _create_embeddings_matrix(self):
        """Crear matriz de embeddings y lista de etiquetas"""
        embeddings = []
        labels = []
        
        for identity_id, identity_data in self.embeddings_by_identity.items():
            for item in identity_data:
                embeddings.append(item['embedding'])
                labels.append(identity_id)
        
        self.embeddings_matrix = np.array(embeddings)
        self.identity_labels = np.array(labels)
        
        logger.info(f"Matriz de embeddings: {self.embeddings_matrix.shape}")
    
    def calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calcular similitud coseno entre dos embeddings
        
        Args:
            embedding1, embedding2: Vectores de embedding
            
        Returns:
            Similitud coseno (0-1)
        """
        # Normalizar embeddings
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Calcular similitud coseno
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)
    
    def calculate_euclidean_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calcular distancia euclidiana entre dos embeddings
        
        Args:
            embedding1, embedding2: Vectores de embedding
            
        Returns:
            Distancia euclidiana
        """
        return float(np.linalg.norm(embedding1 - embedding2))
    
    def verify_identity(self, 
                       embedding1: np.ndarray, 
                       embedding2: np.ndarray,
                       threshold: float = 0.35,
                       metric: str = "cosine") -> Dict:
        """
        Verificar si dos embeddings corresponden a la misma identidad
        
        Args:
            embedding1, embedding2: Vectores de embedding
            threshold: Umbral de decisión
            metric: Métrica a usar ("cosine" o "euclidean")
            
        Returns:
            Diccionario con resultado de verificación
        """
        if metric == "cosine":
            similarity = self.calculate_cosine_similarity(embedding1, embedding2)
            is_match = similarity >= threshold
            score = similarity
        elif metric == "euclidean":
            distance = self.calculate_euclidean_distance(embedding1, embedding2)
            is_match = distance <= threshold
            score = distance
        else:
            raise ValueError(f"Métrica no soportada: {metric}")
        
        return {
            'is_match': is_match,
            'score': score,
            'threshold': threshold,
            'metric': metric
        }
    
    def find_most_similar(self, 
                         query_embedding: np.ndarray, 
                         top_k: int = 5,
                         exclude_identity: Optional[str] = None) -> List[Dict]:
        """
        Encontrar los embeddings más similares a una consulta
        
        Args:
            query_embedding: Embedding de consulta
            top_k: Número de resultados más similares
            exclude_identity: Identidad a excluir de los resultados
            
        Returns:
            Lista de resultados ordenados por similitud
        """
        if self.embeddings_matrix is None:
            raise ValueError("No hay embeddings cargados")
        
        # Calcular similitudes
        similarities = []
        for i, embedding in enumerate(self.embeddings_matrix):
            identity = self.identity_labels[i]
            
            # Excluir identidad si se especifica
            if exclude_identity and identity == exclude_identity:
                continue
            
            similarity = self.calculate_cosine_similarity(query_embedding, embedding)
            similarities.append({
                'index': i,
                'identity': identity,
                'similarity': similarity
            })
        
        # Ordenar por similitud descendente
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k]
    
    def calculate_intra_identity_statistics(self) -> Dict:
        """
        Calcular estadísticas intra-identidad (variabilidad dentro de cada identidad)
        
        Returns:
            Diccionario con estadísticas por identidad
        """
        stats = {}
        
        for identity_id, identity_data in self.embeddings_by_identity.items():
            if len(identity_data) < 2:
                continue  # Necesitamos al menos 2 muestras
            
            # Obtener embeddings de la identidad
            embeddings = [item['embedding'] for item in identity_data]
            embeddings = np.array(embeddings)
            
            # Calcular matriz de similitudes intra-identidad
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = self.calculate_cosine_similarity(embeddings[i], embeddings[j])
                    similarities.append(sim)
            
            similarities = np.array(similarities)
            
            stats[identity_id] = {
                'num_samples': len(embeddings),
                'mean_similarity': np.mean(similarities),
                'std_similarity': np.std(similarities),
                'min_similarity': np.min(similarities),
                'max_similarity': np.max(similarities)
            }
        
        return stats
    
    def calculate_inter_identity_statistics(self, max_comparisons: int = 10000) -> Dict:
        """
        Calcular estadísticas inter-identidad (separabilidad entre identidades)
        
        Args:
            max_comparisons: Máximo número de comparaciones para evitar sobrecarga
            
        Returns:
            Diccionario con estadísticas inter-identidad
        """
        if self.embeddings_matrix is None:
            raise ValueError("No hay embeddings cargados")
        
        # Muestrear pares de diferentes identidades
        different_identity_similarities = []
        comparisons = 0
        
        unique_identities = list(set(self.identity_labels))
        
        for i, identity1 in enumerate(unique_identities):
            for j, identity2 in enumerate(unique_identities[i+1:], i+1):
                if comparisons >= max_comparisons:
                    break
                
                # Obtener embeddings de cada identidad
                embeddings1 = self.embeddings_matrix[self.identity_labels == identity1]
                embeddings2 = self.embeddings_matrix[self.identity_labels == identity2]
                
                # Comparar primer embedding de cada identidad
                if len(embeddings1) > 0 and len(embeddings2) > 0:
                    sim = self.calculate_cosine_similarity(embeddings1[0], embeddings2[0])
                    different_identity_similarities.append(sim)
                    comparisons += 1
            
            if comparisons >= max_comparisons:
                break
        
        similarities = np.array(different_identity_similarities)
        
        return {
            'num_comparisons': len(similarities),
            'mean_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities)
        }
    
    def evaluate_threshold(self, threshold_range: Tuple[float, float] = (0.2, 0.8), 
                          num_thresholds: int = 50) -> Dict:
        """
        Evaluar diferentes umbrales para verificación de identidad
        
        Args:
            threshold_range: Rango de umbrales a evaluar
            num_thresholds: Número de umbrales a probar
            
        Returns:
            Diccionario con métricas de evaluación por umbral
        """
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
        results = []
        
        # Obtener estadísticas intra e inter identidad
        intra_stats = self.calculate_intra_identity_statistics()
        inter_stats = self.calculate_inter_identity_statistics()
        
        # Simular decisiones para cada umbral
        for threshold in thresholds:
            # Casos positivos verdaderos (misma identidad)
            intra_similarities = []
            for stats in intra_stats.values():
                # Usar similitud media como proxy
                intra_similarities.append(stats['mean_similarity'])
            
            # Casos negativos verdaderos (diferentes identidades)
            inter_similarity = inter_stats['mean_similarity']
            
            # Calcular métricas (simplificado)
            tp = sum(1 for sim in intra_similarities if sim >= threshold)  # True Positives
            fn = sum(1 for sim in intra_similarities if sim < threshold)   # False Negatives
            fp = 1 if inter_similarity >= threshold else 0                # False Positives (simplificado)
            tn = 1 if inter_similarity < threshold else 0                 # True Negatives (simplificado)
            
            # Métricas
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn
            })
        
        return {
            'thresholds': thresholds,
            'results': results,
            'best_f1_threshold': max(results, key=lambda x: x['f1_score'])['threshold']
        }
    
    def visualize_embeddings_2d(self, method: str = "tsne", save_path: Optional[str] = None):
        """
        Visualizar embeddings en 2D usando t-SNE o PCA
        
        Args:
            method: Método de reducción dimensional ("tsne" o "pca")
            save_path: Ruta donde guardar la visualización
        """
        if self.embeddings_matrix is None:
            raise ValueError("No hay embeddings cargados")
        
        # Reducción dimensional
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.embeddings_matrix) - 1))
            embeddings_2d = reducer.fit_transform(self.embeddings_matrix)
        elif method == "pca":
            reducer = PCA(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(self.embeddings_matrix)
        else:
            raise ValueError(f"Método no soportado: {method}")
        
        # Crear visualización
        plt.figure(figsize=(12, 8))
        
        # Obtener identidades únicas y asignar colores
        unique_identities = list(set(self.identity_labels))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_identities)))
        
        for i, identity in enumerate(unique_identities):
            mask = self.identity_labels == identity
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=[colors[i]], label=f'ID {identity}', alpha=0.7, s=50)
        
        plt.title(f'Visualización de Embeddings Faciales ({method.upper()})')
        plt.xlabel('Componente 1')
        plt.ylabel('Componente 2')
        
        # Mostrar leyenda solo si hay pocas identidades
        if len(unique_identities) <= 20:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualización guardada en: {save_path}")
        
        plt.show()
    
    def plot_similarity_distribution(self, save_path: Optional[str] = None):
        """
        Graficar distribución de similitudes intra e inter identidad
        
        Args:
            save_path: Ruta donde guardar el gráfico
        """
        # Obtener similitudes intra-identidad
        intra_similarities = []
        intra_stats = self.calculate_intra_identity_statistics()
        for stats in intra_stats.values():
            intra_similarities.append(stats['mean_similarity'])
        
        # Obtener similitudes inter-identidad
        inter_stats = self.calculate_inter_identity_statistics()
        
        # Crear histograma
        plt.figure(figsize=(10, 6))
        
        # Histograma de similitudes intra-identidad
        plt.hist(intra_similarities, bins=30, alpha=0.7, label='Intra-identidad (misma persona)', 
                color='green', density=True)
        
        # Línea vertical para similitud inter-identidad promedio
        plt.axvline(x=inter_stats['mean_similarity'], color='red', linestyle='--', 
                   label=f'Inter-identidad promedio: {inter_stats["mean_similarity"]:.3f}')
        
        plt.xlabel('Similitud Coseno')
        plt.ylabel('Densidad')
        plt.title('Distribución de Similitudes Intra vs Inter Identidad')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico guardado en: {save_path}")
        
        plt.show()

def main():
    """Función principal para demostrar el uso de las utilidades"""
    
    # Ejemplo de uso
    embeddings_file = "data/celeba/embeddings/celeba_embeddings_full.pkl"
    
    if not Path(embeddings_file).exists():
        print(f"⚠️  Archivo de embeddings no encontrado: {embeddings_file}")
        print("Ejecuta primero celeba_pipeline.py para generar embeddings")
        return
    
    # Inicializar analizador
    analyzer = EmbeddingAnalyzer(embeddings_file)
    
    # Análisis estadístico
    print("📊 Calculando estadísticas...")
    intra_stats = analyzer.calculate_intra_identity_statistics()
    inter_stats = analyzer.calculate_inter_identity_statistics()
    
    print(f"\n🔍 Estadísticas Intra-identidad:")
    print(f"  Identidades con múltiples muestras: {len(intra_stats)}")
    if intra_stats:
        mean_similarities = [stats['mean_similarity'] for stats in intra_stats.values()]
        print(f"  Similitud media intra-identidad: {np.mean(mean_similarities):.3f} ± {np.std(mean_similarities):.3f}")
    
    print(f"\n🔍 Estadísticas Inter-identidad:")
    print(f"  Similitud media inter-identidad: {inter_stats['mean_similarity']:.3f}")
    print(f"  Desviación estándar: {inter_stats['std_similarity']:.3f}")
    
    # Evaluación de umbrales
    print("\n🎯 Evaluando umbrales óptimos...")
    threshold_eval = analyzer.evaluate_threshold()
    best_threshold = threshold_eval['best_f1_threshold']
    print(f"  Mejor umbral (F1-score): {best_threshold:.3f}")
    
    # Visualizaciones
    print("\n📈 Generando visualizaciones...")
    
    # Distribución de similitudes
    analyzer.plot_similarity_distribution("similarity_distribution.png")
    
    # Embeddings en 2D
    analyzer.visualize_embeddings_2d(method="tsne", save_path="embeddings_tsne.png")
    
    print("\n✅ Análisis completado!")

if __name__ == "__main__":
    main()