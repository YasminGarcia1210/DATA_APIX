#!/usr/bin/env python3
"""
Ejemplo de uso del descargador de CelebA
Muestra cómo usar la clase CelebADownloader programáticamente
"""

from download_celeba import CelebADownloader
import logging

# Configurar logging para ver el progreso
logging.basicConfig(level=logging.INFO)

def ejemplo_descarga_completa():
    """Ejemplo: Descargar dataset completo de CelebA"""
    print("🚀 Ejemplo 1: Descarga completa de CelebA")
    
    # Crear descargador
    downloader = CelebADownloader(data_dir="data/celeba")
    
    # Verificar estado actual
    existing = downloader.check_existing_files()
    downloader.print_status(existing)
    
    # Descargar todo (incluye imágenes e identidades)
    success = downloader.download_all(
        include_attributes=True,  # Incluir archivo de atributos
        force=False  # No forzar si ya existen
    )
    
    if success:
        print("✅ Descarga completa exitosa!")
    else:
        print("⚠️  Descarga falló o fue parcial")

def ejemplo_descarga_selectiva():
    """Ejemplo: Descargar solo archivos específicos"""
    print("\n🎯 Ejemplo 2: Descarga selectiva")
    
    downloader = CelebADownloader(data_dir="data/celeba")
    
    # Descargar solo el archivo de identidades
    print("📥 Descargando solo identidades...")
    success = downloader.download_file('identity_celeba', force=False)
    
    if success:
        print("✅ Identidades descargadas")
    
    # Descargar solo las imágenes
    print("📥 Descargando solo imágenes...")
    success = downloader.download_file('img_align_celeba', force=False)
    
    if success:
        print("✅ Imágenes descargadas")
        
        # Extraer automáticamente
        print("📦 Extrayendo imágenes...")
        if downloader.extract_images():
            print("✅ Imágenes extraídas")

def ejemplo_verificar_estado():
    """Ejemplo: Solo verificar qué archivos ya están descargados"""
    print("\n📋 Ejemplo 3: Verificar estado actual")
    
    downloader = CelebADownloader(data_dir="data/celeba")
    
    # Obtener estado actual
    existing = downloader.check_existing_files()
    
    # Mostrar información detallada
    downloader.print_status(existing)
    
    # Verificar si está listo para procesamiento
    img_dir_exists = existing['extracted_images']['exists']
    identity_file_exists = existing['identity_CelebA.txt']['exists']
    
    if img_dir_exists and identity_file_exists:
        img_count = existing['extracted_images']['count']
        print(f"\n🎉 CelebA está listo para usar!")
        print(f"   • {img_count:,} imágenes disponibles")
        print(f"   • Archivo de identidades disponible")
        print(f"\n🚀 Puedes ejecutar: python celeba_pipeline.py")
    else:
        print(f"\n⚠️  CelebA no está completo:")
        if not img_dir_exists:
            print("   • Faltan imágenes extraídas")
        if not identity_file_exists:
            print("   • Falta archivo de identidades")

def ejemplo_uso_personalizado():
    """Ejemplo: Uso con configuración personalizada"""
    print("\n⚙️  Ejemplo 4: Configuración personalizada")
    
    # Crear descargador con directorio personalizado
    custom_dir = "mi_dataset/celeba_custom"
    downloader = CelebADownloader(data_dir=custom_dir)
    
    print(f"📁 Usando directorio personalizado: {custom_dir}")
    
    # Verificar estado
    existing = downloader.check_existing_files()
    downloader.print_status(existing)
    
    # Ejemplo: Solo mostrar instrucciones manuales
    print("\n📖 Instrucciones para descarga manual:")
    downloader.print_manual_instructions()

def main():
    """Ejecutar todos los ejemplos"""
    print("=" * 60)
    print("📚 EJEMPLOS DE USO DEL DESCARGADOR DE CELEBA")
    print("=" * 60)
    
    # Ejemplo 1: Verificar estado (sin descargar)
    ejemplo_verificar_estado()
    
    # Ejemplo 2: Configuración personalizada
    ejemplo_uso_personalizado()
    
    # Los siguientes ejemplos realmente descargan archivos
    # Descomenta solo si quieres ejecutarlos
    
    # ejemplo_descarga_selectiva()
    # ejemplo_descarga_completa()
    
    print("\n" + "=" * 60)
    print("💡 NOTAS:")
    print("   • Para ejecutar descargas reales, descomenta las líneas correspondientes")
    print("   • La descarga completa puede tomar varias horas (1.3GB)")
    print("   • Verifica tu conexión a internet antes de descargar")
    print("=" * 60)

if __name__ == "__main__":
    main()