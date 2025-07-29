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
        self.embeddings_by_identity = {}
        self.embeddings_matrix = None
        self.identity_labels = []
        
        if embeddings_file:
            self.load_embeddings(embeddings_file)

    def load_embeddings(self, embeddings_file: str):
        embeddings_file = Path(embeddings_file)
        if not embeddings_file.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {embeddings_file}")
        with open(embeddings_file, 'rb') as f:
            self.embeddings_by_identity = pickle.load(f)
        self._create_embeddings_matrix()
        logger.info(f"Embeddings cargados: {len(self.embeddings_by_identity)} identidades")

    def _create_embeddings_matrix(self):
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
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)

    def calculate_euclidean_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        return float(np.linalg.norm(embedding1 - embedding2))

    def verify_identity(self, embedding1: np.ndarray, embedding2: np.ndarray, threshold: float = 0.35, metric: str = "cosine") -> Dict:
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
        return {'is_match': is_match, 'score': score, 'threshold': threshold, 'metric': metric}

    def find_most_similar(self, query_embedding: np.ndarray, top_k: int = 5, exclude_identity: Optional[str] = None) -> List[Dict]:
        if self.embeddings_matrix is None:
            raise ValueError("No hay embeddings cargados")
        similarities = []
        for i, embedding in enumerate(self.embeddings_matrix):
            identity = self.identity_labels[i]
            if exclude_identity and identity == exclude_identity:
                continue
            similarity = self.calculate_cosine_similarity(query_embedding, embedding)
            similarities.append({'index': i, 'identity': identity, 'similarity': similarity})
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]

    def calculate_intra_identity_statistics(self) -> Dict:
        stats = {}
        for identity_id, identity_data in self.embeddings_by_identity.items():
            if len(identity_data) < 2:
                continue
            embeddings = [item['embedding'] for item in identity_data]
            embeddings = np.array(embeddings)
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
        if self.embeddings_matrix is None:
            raise ValueError("No hay embeddings cargados")
        different_identity_similarities = []
        comparisons = 0
        unique_identities = list(set(self.identity_labels))
        for i, identity1 in enumerate(unique_identities):
            for j, identity2 in enumerate(unique_identities[i+1:], i+1):
                if comparisons >= max_comparisons:
                    break
                embeddings1 = self.embeddings_matrix[self.identity_labels == identity1]
                embeddings2 = self.embeddings_matrix[self.identity_labels == identity2]
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

    def evaluate_threshold(self, threshold_range: Tuple[float, float] = (0.2, 0.8), num_thresholds: int = 50) -> Dict:
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
        results = []
        intra_stats = self.calculate_intra_identity_statistics()
        inter_stats = self.calculate_inter_identity_statistics()
        for threshold in thresholds:
            intra_similarities = [stats['mean_similarity'] for stats in intra_stats.values()]
            inter_similarity = inter_stats['mean_similarity']
            tp = sum(1 for sim in intra_similarities if sim >= threshold)
            fn = sum(1 for sim in intra_similarities if sim < threshold)
            fp = 1 if inter_similarity >= threshold else 0
            tn = 1 if inter_similarity < threshold else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            results.append({'threshold': threshold, 'precision': precision, 'recall': recall, 'f1_score': f1_score, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn})
        return {
            'thresholds': thresholds,
            'results': results,
            'best_f1_threshold': max(results, key=lambda x: x['f1_score'])['threshold']
        }

    def visualize_embeddings_2d(self, method: str = "tsne", save_path: Optional[str] = None):
        if self.embeddings_matrix is None:
            raise ValueError("No hay embeddings cargados")
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.embeddings_matrix) - 1))
            embeddings_2d = reducer.fit_transform(self.embeddings_matrix)
        elif method == "pca":
            reducer = PCA(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(self.embeddings_matrix)
        else:
            raise ValueError(f"Método no soportado: {method}")
        plt.figure(figsize=(12, 8))
        unique_identities = list(set(self.identity_labels))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_identities)))
        for i, identity in enumerate(unique_identities):
            mask = self.identity_labels == identity
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], c=[colors[i]], label=f'ID {identity}', alpha=0.7, s=50)
        plt.title(f'Visualización de Embeddings Faciales ({method.upper()})')
        plt.xlabel('Componente 1')
        plt.ylabel('Componente 2')
        if len(unique_identities) <= 20:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualización guardada en: {save_path}")
        plt.show()

    def plot_similarity_distribution(self, save_path: Optional[str] = None):
        intra_similarities = []
        intra_stats = self.calculate_intra_identity_statistics()
        for stats in intra_stats.values():
            intra_similarities.append(stats['mean_similarity'])
        inter_stats = self.calculate_inter_identity_statistics()
        plt.figure(figsize=(10, 6))
        plt.hist(intra_similarities, bins=30, alpha=0.7, label='Intra-identidad (misma persona)', color='green', density=True)
        plt.axvline(x=inter_stats['mean_similarity'], color='red', linestyle='--', label=f'Inter-identidad promedio: {inter_stats["mean_similarity"]:.3f}')
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
    embeddings_file = "data/celeba/embeddings/celeba_embeddings_full.pkl"
    if not Path(embeddings_file).exists():
        print(f"⚠️  Archivo de embeddings no encontrado: {embeddings_file}")
        print("Ejecuta primero celeba_pipeline.py para generar embeddings")
        return
    analyzer = EmbeddingAnalyzer(embeddings_file)
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
    print("\n🎯 Evaluando umbrales óptimos...")
    threshold_eval = analyzer.evaluate_threshold()
    best_threshold = threshold_eval['best_f1_threshold']
    print(f"  Mejor umbral (F1-score): {best_threshold:.3f}")
    print("\n📈 Generando visualizaciones...")
    analyzer.plot_similarity_distribution("similarity_distribution.png")
    analyzer.visualize_embeddings_2d(method="tsne", save_path="embeddings_tsne.png")
    print("\n✅ Análisis completado!")

if __name__ == "__main__":
    main()
