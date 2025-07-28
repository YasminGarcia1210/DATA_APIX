#!/usr/bin/env python3
"""
Script maestro para ejecutar todos los tests del pipeline de verificación facial
Incluye testing de embeddings, verificación y generación de reportes
"""

import time
import json
from pathlib import Path
from datetime import datetime
import logging

# Importar módulos de testing
from test_embeddings import EmbeddingTester, test_sample_images, test_celeba_sample, test_model_comparison
from test_verification import VerificationTester, test_verification_with_celeba, test_verification_with_samples
from embedding_utils import EmbeddingAnalyzer

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestSuite:
    """Suite completa de tests para el pipeline facial"""
    
    def __init__(self, output_dir: str = "test_results"):
        """
        Inicializar suite de tests
        
        Args:
            output_dir: Directorio para guardar resultados
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Resultados de todos los tests
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'embedding_tests': {},
            'verification_tests': {},
            'performance_summary': {},
            'errors': []
        }
        
        self.start_time = None
        self.total_time = 0
    
    def log_error(self, test_name: str, error: str):
        """Registrar error de test"""
        error_entry = {
            'test': test_name,
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        }
        self.test_results['errors'].append(error_entry)
        logger.error(f"❌ Error en {test_name}: {error}")
    
    def run_embedding_tests(self) -> dict:
        """Ejecutar todos los tests de embeddings"""
        logger.info("🧠 Iniciando tests de embeddings...")
        embedding_results = {}
        
        # Test 1: Imágenes de muestra
        try:
            logger.info("Test 1: Embeddings con imágenes de muestra")
            sample_results = test_sample_images()
            embedding_results['sample_images'] = {
                'success': True,
                'results_count': len(sample_results) if sample_results else 0
            }
        except Exception as e:
            self.log_error("embedding_sample_images", e)
            embedding_results['sample_images'] = {'success': False, 'error': str(e)}
        
        # Test 2: Muestra de CelebA
        try:
            logger.info("Test 2: Embeddings con CelebA")
            celeba_results = test_celeba_sample()
            embedding_results['celeba_sample'] = {
                'success': True,
                'results_count': len(celeba_results) if celeba_results else 0
            }
        except Exception as e:
            self.log_error("embedding_celeba_sample", e)
            embedding_results['celeba_sample'] = {'success': False, 'error': str(e)}
        
        # Test 3: Comparación de modelos
        try:
            logger.info("Test 3: Comparación de modelos")
            test_model_comparison()
            embedding_results['model_comparison'] = {'success': True}
        except Exception as e:
            self.log_error("embedding_model_comparison", e)
            embedding_results['model_comparison'] = {'success': False, 'error': str(e)}
        
        # Test 4: Benchmark detallado
        try:
            logger.info("Test 4: Benchmark detallado de embeddings")
            benchmark_results = self.run_embedding_benchmark()
            embedding_results['detailed_benchmark'] = benchmark_results
        except Exception as e:
            self.log_error("embedding_detailed_benchmark", e)
            embedding_results['detailed_benchmark'] = {'success': False, 'error': str(e)}
        
        return embedding_results
    
    def run_verification_tests(self) -> dict:
        """Ejecutar todos los tests de verificación"""
        logger.info("🔍 Iniciando tests de verificación...")
        verification_results = {}
        
        # Test 1: Verificación con imágenes de muestra
        try:
            logger.info("Test 1: Verificación con imágenes de muestra")
            sample_tester = test_verification_with_samples()
            if sample_tester:
                metrics = sample_tester.calculate_metrics()
                verification_results['sample_images'] = {
                    'success': True,
                    'metrics': metrics
                }
            else:
                verification_results['sample_images'] = {'success': False, 'error': 'No se pudo crear tester'}
        except Exception as e:
            self.log_error("verification_sample_images", e)
            verification_results['sample_images'] = {'success': False, 'error': str(e)}
        
        # Test 2: Verificación con CelebA
        try:
            logger.info("Test 2: Verificación con CelebA")
            celeba_tester = test_verification_with_celeba()
            if celeba_tester:
                metrics = celeba_tester.calculate_metrics()
                verification_results['celeba'] = {
                    'success': True,
                    'metrics': metrics
                }
            else:
                verification_results['celeba'] = {'success': False, 'error': 'Dataset no disponible'}
        except Exception as e:
            self.log_error("verification_celeba", e)
            verification_results['celeba'] = {'success': False, 'error': str(e)}
        
        # Test 3: Análisis de umbrales
        try:
            logger.info("Test 3: Análisis de umbrales óptimos")
            threshold_results = self.run_threshold_analysis()
            verification_results['threshold_analysis'] = threshold_results
        except Exception as e:
            self.log_error("verification_threshold_analysis", e)
            verification_results['threshold_analysis'] = {'success': False, 'error': str(e)}
        
        return verification_results
    
    def run_embedding_benchmark(self) -> dict:
        """Benchmark detallado de embeddings"""
        logger.info("🏁 Ejecutando benchmark detallado...")
        
        # Buscar imágenes para benchmark
        test_images = []
        
        # Imágenes de muestra
        sample_dir = Path("data/sample_images")
        if sample_dir.exists():
            test_images.extend([str(f) for f in sample_dir.glob("*.jpg")][:5])
        
        # CelebA
        celeba_dir = Path("data/celeba/img_align_celeba")
        if celeba_dir.exists():
            test_images.extend([str(f) for f in celeba_dir.glob("*.jpg")][:10])
        
        if not test_images:
            return {'success': False, 'error': 'No hay imágenes para benchmark'}
        
        models_to_test = ['buffalo_l']
        
        # Verificar si arcface_r50 está disponible
        try:
            from celeba_pipeline import CelebAProcessor
            test_processor = CelebAProcessor(model_name='arcface_r50')
            models_to_test.append('arcface_r50')
            logger.info("✅ Modelo arcface_r50 disponible")
        except Exception:
            logger.info("⚠️  Modelo arcface_r50 no disponible")
        
        benchmark_results = {}
        
        for model in models_to_test:
            logger.info(f"📊 Benchmarking modelo: {model}")
            
            try:
                tester = EmbeddingTester(model_name=model)
                
                start_time = time.time()
                results = tester.test_multiple_images(test_images[:5])  # Usar solo 5 para benchmark
                benchmark_time = time.time() - start_time
                
                tester.calculate_performance_stats()
                
                # Analizar calidad de embeddings
                successful_results = [r for r in results if r['success']]
                if successful_results:
                    embeddings = [r['embedding'] for r in successful_results]
                    embedding_analysis = self.analyze_embedding_quality(embeddings)
                else:
                    embedding_analysis = {}
                
                benchmark_results[model] = {
                    'success': True,
                    'total_time': benchmark_time,
                    'performance_metrics': tester.performance_metrics,
                    'embedding_analysis': embedding_analysis,
                    'images_processed': len(successful_results)
                }
                
            except Exception as e:
                benchmark_results[model] = {
                    'success': False,
                    'error': str(e)
                }
        
        return benchmark_results
    
    def analyze_embedding_quality(self, embeddings: list) -> dict:
        """Analizar calidad de una lista de embeddings"""
        import numpy as np
        
        embeddings = np.array(embeddings)
        
        # Estadísticas básicas
        mean_embedding = np.mean(embeddings, axis=0)
        std_embedding = np.std(embeddings, axis=0)
        
        # Diversidad de embeddings
        pairwise_similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                pairwise_similarities.append(sim)
        
        return {
            'num_embeddings': len(embeddings),
            'embedding_dimension': embeddings.shape[1],
            'mean_l2_norm': float(np.mean([np.linalg.norm(emb) for emb in embeddings])),
            'std_l2_norm': float(np.std([np.linalg.norm(emb) for emb in embeddings])),
            'mean_pairwise_similarity': float(np.mean(pairwise_similarities)) if pairwise_similarities else 0,
            'std_pairwise_similarity': float(np.std(pairwise_similarities)) if pairwise_similarities else 0,
            'embedding_range': {
                'min': float(np.min(embeddings)),
                'max': float(np.max(embeddings)),
                'mean': float(np.mean(embeddings)),
                'std': float(np.std(embeddings))
            }
        }
    
    def run_threshold_analysis(self) -> dict:
        """Análisis detallado de umbrales óptimos"""
        logger.info("📊 Analizando umbrales óptimos...")
        
        # Usar embeddings previamente guardados si están disponibles
        embeddings_file = Path("data/celeba/embeddings/celeba_embeddings_full.pkl")
        
        if not embeddings_file.exists():
            return {'success': False, 'error': 'No hay embeddings guardados para análisis'}
        
        try:
            analyzer = EmbeddingAnalyzer(str(embeddings_file))
            
            # Calcular estadísticas
            intra_stats = analyzer.calculate_intra_identity_statistics()
            inter_stats = analyzer.calculate_inter_identity_statistics()
            
            # Evaluar umbrales
            threshold_eval = analyzer.evaluate_threshold()
            
            return {
                'success': True,
                'intra_identity_stats': {
                    'num_identities_analyzed': len(intra_stats),
                    'avg_intra_similarity': float(np.mean([s['mean_similarity'] for s in intra_stats.values()])) if intra_stats else 0
                },
                'inter_identity_stats': inter_stats,
                'optimal_threshold': threshold_eval['best_f1_threshold'],
                'threshold_evaluation': {
                    'num_thresholds_tested': len(threshold_eval['results']),
                    'best_f1_score': max([r['f1_score'] for r in threshold_eval['results']]) if threshold_eval['results'] else 0
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def calculate_performance_summary(self) -> dict:
        """Calcular resumen general de rendimiento"""
        summary = {
            'total_execution_time': self.total_time,
            'tests_executed': 0,
            'tests_successful': 0,
            'tests_failed': 0,
            'error_count': len(self.test_results['errors'])
        }
        
        # Contar tests de embeddings
        for test_name, result in self.test_results['embedding_tests'].items():
            summary['tests_executed'] += 1
            if isinstance(result, dict) and result.get('success', False):
                summary['tests_successful'] += 1
            else:
                summary['tests_failed'] += 1
        
        # Contar tests de verificación
        for test_name, result in self.test_results['verification_tests'].items():
            summary['tests_executed'] += 1
            if isinstance(result, dict) and result.get('success', False):
                summary['tests_successful'] += 1
            else:
                summary['tests_failed'] += 1
        
        # Calcular tasa de éxito
        if summary['tests_executed'] > 0:
            summary['success_rate'] = summary['tests_successful'] / summary['tests_executed']
        else:
            summary['success_rate'] = 0
        
        return summary
    
    def generate_report(self):
        """Generar reporte completo de tests"""
        logger.info("📄 Generando reporte completo...")
        
        # Calcular resumen de rendimiento
        self.test_results['performance_summary'] = self.calculate_performance_summary()
        
        # Guardar reporte JSON
        report_file = self.output_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        # Generar reporte HTML
        html_report = self.generate_html_report()
        html_file = self.output_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        logger.info(f"📊 Reporte JSON guardado: {report_file}")
        logger.info(f"🌐 Reporte HTML guardado: {html_file}")
        
        # Mostrar resumen en consola
        self.print_summary()
    
    def generate_html_report(self) -> str:
        """Generar reporte HTML"""
        summary = self.test_results['performance_summary']
        
        html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte de Tests - Pipeline Facial</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .summary {{ background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #3498db; color: white; border-radius: 5px; min-width: 150px; text-align: center; }}
        .success {{ background: #27ae60; }}
        .error {{ background: #e74c3c; }}
        .warning {{ background: #f39c12; }}
        .test-section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #3498db; background: #f8f9fa; }}
        .error-list {{ background: #fdeaea; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        pre {{ background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Reporte de Tests - Pipeline de Verificación Facial</h1>
        
        <div class="summary">
            <h2>📊 Resumen General</h2>
            <div class="metric success">
                <div><strong>{summary['tests_successful']}</strong></div>
                <div>Tests Exitosos</div>
            </div>
            <div class="metric error">
                <div><strong>{summary['tests_failed']}</strong></div>
                <div>Tests Fallidos</div>
            </div>
            <div class="metric">
                <div><strong>{summary['success_rate']:.1%}</strong></div>
                <div>Tasa de Éxito</div>
            </div>
            <div class="metric warning">
                <div><strong>{summary['total_execution_time']:.1f}s</strong></div>
                <div>Tiempo Total</div>
            </div>
        </div>
        
        <div class="test-section">
            <h2>🧠 Tests de Embeddings</h2>
            <pre>{json.dumps(self.test_results['embedding_tests'], indent=2, default=str)}</pre>
        </div>
        
        <div class="test-section">
            <h2>🔍 Tests de Verificación</h2>
            <pre>{json.dumps(self.test_results['verification_tests'], indent=2, default=str)}</pre>
        </div>
        
        {self.generate_error_section()}
        
        <div class="timestamp">
            <p><strong>Generado:</strong> {self.test_results['timestamp']}</p>
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def generate_error_section(self) -> str:
        """Generar sección de errores para el HTML"""
        if not self.test_results['errors']:
            return '<div class="test-section"><h2>✅ Sin Errores</h2><p>Todos los tests se ejecutaron sin errores.</p></div>'
        
        error_html = '<div class="test-section"><h2>❌ Errores Encontrados</h2>'
        for error in self.test_results['errors']:
            error_html += f'''
            <div class="error-list">
                <strong>Test:</strong> {error['test']}<br>
                <strong>Error:</strong> {error['error']}<br>
                <span class="timestamp">Timestamp: {error['timestamp']}</span>
            </div>
            '''
        error_html += '</div>'
        return error_html
    
    def print_summary(self):
        """Mostrar resumen en consola"""
        summary = self.test_results['performance_summary']
        
        print(f"\n{'='*80}")
        print(f"🎯 RESUMEN FINAL DE TESTS")
        print(f"{'='*80}")
        print(f"⏱️  Tiempo total de ejecución: {summary['total_execution_time']:.2f} segundos")
        print(f"📊 Tests ejecutados: {summary['tests_executed']}")
        print(f"✅ Tests exitosos: {summary['tests_successful']}")
        print(f"❌ Tests fallidos: {summary['tests_failed']}")
        print(f"📈 Tasa de éxito: {summary['success_rate']:.1%}")
        print(f"🐛 Errores registrados: {summary['error_count']}")
        
        if self.test_results['errors']:
            print(f"\n🔍 Errores encontrados:")
            for error in self.test_results['errors']:
                print(f"  • {error['test']}: {error['error']}")
        
        print(f"\n📁 Reportes guardados en: {self.output_dir}")
        print(f"{'='*80}")
    
    def run_all_tests(self):
        """Ejecutar suite completa de tests"""
        logger.info("🚀 Iniciando suite completa de tests...")
        self.start_time = time.time()
        
        try:
            # Tests de embeddings
            self.test_results['embedding_tests'] = self.run_embedding_tests()
            
            # Tests de verificación
            self.test_results['verification_tests'] = self.run_verification_tests()
            
        except Exception as e:
            self.log_error("suite_execution", e)
        
        finally:
            self.total_time = time.time() - self.start_time
            self.generate_report()

def main():
    """Función principal"""
    print("🧪 SUITE COMPLETA DE TESTS - PIPELINE FACIAL")
    print("="*60)
    
    # Crear y ejecutar suite de tests
    test_suite = TestSuite()
    test_suite.run_all_tests()
    
    print("\n🎉 Suite de tests completada!")

if __name__ == "__main__":
    main()