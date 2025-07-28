#!/usr/bin/env python3
"""
Script para descargar automáticamente el dataset CelebA
Soporta múltiples fuentes: Google Drive, Kaggle, descarga manual
"""

import os
import sys
import zipfile
import tarfile
import requests
import gdown
from pathlib import Path
from tqdm import tqdm
import logging
import argparse
import subprocess

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CelebADownloader:
    """Descargador automático del dataset CelebA"""
    
    def __init__(self, data_dir: str = "data/celeba"):
        """
        Inicializar descargador
        
        Args:
            data_dir: Directorio donde guardar CelebA
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # URLs oficiales y alternativos
        self.urls = {
            'img_align_celeba': {
                'google_drive': '0B7EVK8r0v71pZjFTYXZWM3FlRnM',  # URL oficial
                'kaggle': 'jessicali9530/celeba-dataset',
                'alternative': 'https://www.dropbox.com/s/d1kjpkqklf0uw77/img_align_celeba.zip?dl=1'
            },
            'identity_celeba': {
                'google_drive': '1_ee_0u7MG9p55c7rD96ymNeYb2UGQ9m',  # URL oficial
                'direct': 'https://www.dropbox.com/s/h5ebhae8mm4qk3k/identity_CelebA.txt?dl=1'
            },
            'attr_celeba': {
                'google_drive': '0B7EVK8r0v71pblRyaVFSWGxPY0U',  # Atributos (opcional)
                'direct': 'https://www.dropbox.com/s/auexdy98c6g7y9s/list_attr_celeba.txt?dl=1'
            }
        }
        
        # Archivos esperados
        self.expected_files = {
            'img_align_celeba.zip': 'Imágenes alineadas (1.3GB)',
            'identity_CelebA.txt': 'Identidades (~400KB)', 
            'list_attr_celeba.txt': 'Atributos (~13MB)'
        }
    
    def check_existing_files(self) -> dict:
        """Verificar qué archivos ya existen"""
        existing = {}
        
        # Verificar archivos descargados
        for filename, description in self.expected_files.items():
            file_path = self.data_dir / filename
            existing[filename] = {
                'exists': file_path.exists(),
                'path': file_path,
                'size': file_path.stat().st_size if file_path.exists() else 0,
                'description': description
            }
        
        # Verificar imágenes extraídas
        img_dir = self.data_dir / "img_align_celeba"
        existing['extracted_images'] = {
            'exists': img_dir.exists() and len(list(img_dir.glob("*.jpg"))) > 0,
            'path': img_dir,
            'count': len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
        }
        
        return existing
    
    def download_with_progress(self, url: str, filename: str) -> bool:
        """Descargar archivo con barra de progreso"""
        file_path = self.data_dir / filename
        
        try:
            logger.info(f"Descargando {filename} desde {url}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(file_path, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    progress_bar.update(size)
            
            logger.info(f"✅ Descargado: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error descargando {filename}: {e}")
            if file_path.exists():
                file_path.unlink()  # Eliminar archivo parcial
            return False
    
    def download_from_google_drive(self, file_id: str, filename: str) -> bool:
        """Descargar desde Google Drive usando gdown"""
        file_path = self.data_dir / filename
        
        try:
            logger.info(f"Descargando {filename} desde Google Drive (ID: {file_id})")
            
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(file_path), quiet=False)
            
            if file_path.exists() and file_path.stat().st_size > 1000:
                logger.info(f"✅ Descargado desde Google Drive: {filename}")
                return True
            else:
                logger.error(f"❌ Descarga fallida o archivo muy pequeño: {filename}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error con Google Drive para {filename}: {e}")
            return False
    
    def download_from_kaggle(self, dataset: str, filename: str = None) -> bool:
        """Descargar desde Kaggle (requiere kaggle CLI)"""
        try:
            # Verificar si kaggle está instalado
            result = subprocess.run(['kaggle', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning("⚠️  Kaggle CLI no está instalado")
                return False
            
            logger.info(f"Descargando dataset de Kaggle: {dataset}")
            
            # Descargar dataset completo
            cmd = ['kaggle', 'datasets', 'download', '-d', dataset, '-p', str(self.data_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Descargado desde Kaggle")
                return True
            else:
                logger.error(f"❌ Error con Kaggle: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error con Kaggle: {e}")
            return False
    
    def extract_images(self) -> bool:
        """Extraer imágenes del archivo ZIP"""
        zip_path = self.data_dir / "img_align_celeba.zip"
        extract_dir = self.data_dir / "img_align_celeba"
        
        if not zip_path.exists():
            logger.error(f"❌ Archivo ZIP no encontrado: {zip_path}")
            return False
        
        try:
            logger.info(f"Extrayendo imágenes a {extract_dir}")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extraer con barra de progreso
                members = zip_ref.infolist()
                
                with tqdm(total=len(members), desc="Extrayendo") as progress_bar:
                    for member in members:
                        zip_ref.extract(member, self.data_dir)
                        progress_bar.update(1)
            
            # Verificar extracción
            if extract_dir.exists():
                image_count = len(list(extract_dir.glob("*.jpg")))
                logger.info(f"✅ Extraídas {image_count} imágenes")
                return True
            else:
                logger.error("❌ La extracción falló")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error extrayendo imágenes: {e}")
            return False
    
    def download_file(self, file_key: str, force: bool = False) -> bool:
        """
        Descargar un archivo específico con múltiples fuentes de respaldo
        
        Args:
            file_key: Clave del archivo ('img_align_celeba', 'identity_celeba', etc.)
            force: Forzar descarga aunque el archivo ya exista
        """
        if file_key not in self.urls:
            logger.error(f"❌ Archivo no reconocido: {file_key}")
            return False
        
        # Determinar nombre del archivo
        if file_key == 'img_align_celeba':
            filename = 'img_align_celeba.zip'
        elif file_key == 'identity_celeba':
            filename = 'identity_CelebA.txt'
        elif file_key == 'attr_celeba':
            filename = 'list_attr_celeba.txt'
        else:
            filename = f"{file_key}.txt"
        
        file_path = self.data_dir / filename
        
        # Verificar si ya existe
        if file_path.exists() and not force:
            logger.info(f"✅ Archivo ya existe: {filename}")
            return True
        
        urls_to_try = self.urls[file_key]
        success = False
        
        # Intentar Google Drive primero
        if 'google_drive' in urls_to_try:
            success = self.download_from_google_drive(
                urls_to_try['google_drive'], filename
            )
        
        # Intentar Kaggle si Google Drive falla
        if not success and 'kaggle' in urls_to_try:
            success = self.download_from_kaggle(urls_to_try['kaggle'], filename)
        
        # Intentar URL directa/alternativa
        if not success:
            for url_type in ['direct', 'alternative']:
                if url_type in urls_to_try:
                    success = self.download_with_progress(
                        urls_to_try[url_type], filename
                    )
                    if success:
                        break
        
        return success
    
    def download_all(self, include_attributes: bool = False, force: bool = False) -> bool:
        """
        Descargar todo el dataset CelebA
        
        Args:
            include_attributes: Si incluir archivo de atributos
            force: Forzar descarga aunque los archivos ya existan
        """
        logger.info("🚀 Iniciando descarga del dataset CelebA")
        
        # Mostrar estado actual
        existing = self.check_existing_files()
        self.print_status(existing)
        
        success_count = 0
        total_downloads = 0
        
        # Descargar imágenes
        total_downloads += 1
        if self.download_file('img_align_celeba', force):
            success_count += 1
            
            # Extraer imágenes si es necesario
            if not existing['extracted_images']['exists'] or force:
                if self.extract_images():
                    logger.info("✅ Imágenes extraídas correctamente")
                else:
                    logger.warning("⚠️  Problemas extrayendo imágenes")
        
        # Descargar identidades
        total_downloads += 1
        if self.download_file('identity_celeba', force):
            success_count += 1
        
        # Descargar atributos si se solicita
        if include_attributes:
            total_downloads += 1
            if self.download_file('attr_celeba', force):
                success_count += 1
        
        # Mostrar resultados
        logger.info(f"📊 Descarga completada: {success_count}/{total_downloads} archivos")
        
        if success_count == total_downloads:
            logger.info("🎉 ¡Descarga completa exitosa!")
            self.print_next_steps()
            return True
        else:
            logger.warning("⚠️  Descarga parcial. Algunos archivos fallaron.")
            self.print_manual_instructions()
            return False
    
    def print_status(self, existing: dict):
        """Mostrar estado actual de los archivos"""
        print(f"\n📋 Estado actual de CelebA en {self.data_dir}:")
        print("=" * 60)
        
        for filename, info in existing.items():
            if filename == 'extracted_images':
                status = "✅" if info['exists'] else "❌"
                count = info['count'] if info['exists'] else 0
                print(f"{status} Imágenes extraídas: {count:,} archivos")
            else:
                status = "✅" if info['exists'] else "❌"
                size = f"({info['size'] / (1024**2):.1f} MB)" if info['exists'] else ""
                print(f"{status} {filename}: {info['description']} {size}")
        
        print("=" * 60)
    
    def print_next_steps(self):
        """Mostrar próximos pasos después de la descarga"""
        img_dir = self.data_dir / "img_align_celeba"
        img_count = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
        
        print(f"""
🎯 ¡CelebA descargado exitosamente!

📂 Archivos en {self.data_dir}:
   • img_align_celeba/     → {img_count:,} imágenes
   • identity_CelebA.txt   → Metadatos de identidades
   
🚀 Próximos pasos:
   1. Procesar dataset:
      python celeba_pipeline.py
      
   2. Ver demo:
      python demo_pipeline.py
      
   3. Analizar embeddings:
      python embedding_utils.py
""")
    
    def print_manual_instructions(self):
        """Mostrar instrucciones para descarga manual"""
        print(f"""
📥 DESCARGA MANUAL DE CELEBA

Si la descarga automática falló, descarga manualmente desde:
🌐 http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

📁 Archivos necesarios:
   1. "Align&Cropped Images" → Guardar como: {self.data_dir}/img_align_celeba.zip
   2. "Identity CelebA"      → Guardar como: {self.data_dir}/identity_CelebA.txt
   3. (Opcional) "Attr CelebA" → Guardar como: {self.data_dir}/list_attr_celeba.txt

💡 Después de descargar:
   1. Extraer img_align_celeba.zip en {self.data_dir}/
   2. Ejecutar: python celeba_pipeline.py
""")

def main():
    """Función principal con argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description="Descargar dataset CelebA")
    parser.add_argument("--data-dir", default="data/celeba", 
                       help="Directorio donde guardar CelebA")
    parser.add_argument("--include-attr", action="store_true",
                       help="Incluir archivo de atributos")
    parser.add_argument("--force", action="store_true",
                       help="Forzar descarga aunque los archivos ya existan")
    parser.add_argument("--file", choices=['images', 'identity', 'attributes'],
                       help="Descargar solo un archivo específico")
    parser.add_argument("--status", action="store_true",
                       help="Solo mostrar estado actual sin descargar")
    
    args = parser.parse_args()
    
    # Crear descargador
    downloader = CelebADownloader(args.data_dir)
    
    # Solo mostrar estado
    if args.status:
        existing = downloader.check_existing_files()
        downloader.print_status(existing)
        return
    
    # Descargar archivo específico
    if args.file:
        file_map = {
            'images': 'img_align_celeba',
            'identity': 'identity_celeba', 
            'attributes': 'attr_celeba'
        }
        
        success = downloader.download_file(file_map[args.file], args.force)
        
        if args.file == 'images' and success:
            downloader.extract_images()
        
        return
    
    # Descarga completa
    success = downloader.download_all(
        include_attributes=args.include_attr,
        force=args.force
    )
    
    if success:
        print("\n🎉 ¡Listo para usar CelebA con el pipeline!")
    else:
        print("\n⚠️  Revisa las instrucciones para descarga manual")

if __name__ == "__main__":
    main()