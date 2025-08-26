#!/usr/bin/env python3
"""
Flujo Spark con Prefect - Versión corregida con configuración de conexión
"""

import os
import subprocess
import sys
from pyspark.sql import SparkSession
from prefect.blocks.system import Secret
import py7zr
import tempfile
import shutil

class Flujo7zSecretPrefFixed:
    def __init__(self):
        self.secret_block_name = "photos-7z-password"
        self.spark_session = None
        
    def configure_prefect_connection(self):
        """Configura la conexión a Prefect en el contenedor Docker"""
        print("🔧 Configurando conexión a Prefect...")
        
        # Intentar obtener la IP del host desde el contenedor Docker
        try:
            result = subprocess.run(['ip', 'route', 'show', 'default'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gateway_ip = result.stdout.split()[2]
                host_ip = gateway_ip
            else:
                host_ip = "172.17.0.1"  # IP común del gateway Docker
        except Exception:
            host_ip = "172.17.0.1"
        
        # Configurar URL de la API de Prefect
        api_url = f"http://{host_ip}:4200/api"
        print(f"Configurando Prefect API URL a: {api_url}")
        
        # Establecer variable de entorno
        os.environ['PREFECT_API_URL'] = api_url
        
        # También configurar usando prefect config
        try:
            subprocess.run(['prefect', 'config', 'set', f'PREFECT_API_URL={api_url}'], 
                          check=True, capture_output=True)
            print("✅ URL de API de Prefect configurada exitosamente")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error configurando Prefect: {e}")
            return False
    
    def test_prefect_connection(self):
        """Prueba la conexión a Prefect y lista los bloques disponibles"""
        try:
            print("Probando conexión a Prefect...")
            result = subprocess.run(['prefect', 'block', 'ls'], 
                                  capture_output=True, text=True, check=True)
            print("✅ Conexión exitosa al servidor Prefect")
            print("Bloques disponibles:")
            print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Falló la conexión al servidor Prefect: {e}")
            return False

    def load_password(self):
        """Carga la contraseña desde el bloque Secret de Prefect"""
        try:
            print(f"Cargando contraseña desde el bloque: {self.secret_block_name}")
            secret_block = Secret.load(self.secret_block_name)
            password = secret_block.get()
            print("✅ Contraseña cargada exitosamente desde Prefect")
            return password
        except Exception as e:
            print(f"❌ Error cargando la contraseña: {e}")
            raise

    def init_spark(self):
        """Inicializa la sesión de Spark"""
        print("Inicializando sesión de Spark...")
        self.spark_session = SparkSession.builder \
            .appName("Flujo7zSecretPrefFixed") \
            .config("spark.sql.catalog.lake", "org.apache.iceberg.spark.SparkCatalog") \
            .config("spark.sql.catalog.lake.type", "hadoop") \
            .config("spark.sql.catalog.lake.warehouse", "s3a://warehouse/iceberg/") \
            .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
            .getOrCreate()
        print("✅ Sesión de Spark inicializada")

    def process_7z_file(self, file_path, password):
        """Procesa archivo 7z con la contraseña obtenida"""
        print(f"Procesando archivo 7z: {file_path}")
        
        # Crear directorio temporal
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Extraer archivo 7z
                with py7zr.SevenZipFile(file_path, mode='r', password=password) as archive:
                    archive.extractall(temp_dir)
                    print(f"✅ Archivo extraído exitosamente en: {temp_dir}")
                    
                    # Listar archivos extraídos
                    extracted_files = os.listdir(temp_dir)
                    print(f"Archivos extraídos: {extracted_files}")
                    
                    # Aquí puedes agregar la lógica de procesamiento de los archivos extraídos
                    # Por ejemplo, cargar como DataFrame de Spark, procesar con Iceberg, etc.
                    
                    return True
                    
            except py7zr.exceptions.Bad7zFile as e:
                print(f"❌ Error: Archivo 7z corrupto o inválido: {e}")
                return False
            except Exception as e:
                print(f"❌ Error procesando archivo 7z: {e}")
                return False

    def create_sample_data(self):
        """Crea datos de ejemplo para demostrar el funcionamiento"""
        print("Creando datos de ejemplo...")
        
        # Crear DataFrame de ejemplo
        data = [
            ("archivo1.txt", "contenido1", "2024-01-01"),
            ("archivo2.txt", "contenido2", "2024-01-02"),
            ("archivo3.txt", "contenido3", "2024-01-03")
        ]
        
        df = self.spark_session.createDataFrame(data, ["filename", "content", "date"])
        df.show()
        
        # Aquí podrías guardar en Iceberg o procesar según tus necesidades
        print("✅ Datos de ejemplo creados y mostrados")
        
        return df

    def cleanup(self):
        """Limpia recursos"""
        if self.spark_session:
            self.spark_session.stop()
            print("✅ Sesión de Spark cerrada")

    def run(self):
        """Ejecuta el flujo principal"""
        try:
            print("🚀 Iniciando Flujo7zSecretPrefFixed...")
            
            # 1. Configurar conexión a Prefect
            if not self.configure_prefect_connection():
                raise Exception("No se pudo configurar la conexión a Prefect")
            
            # 2. Probar conexión a Prefect
            if not self.test_prefect_connection():
                raise Exception("No se pudo conectar al servidor Prefect")
            
            # 3. Cargar contraseña desde Prefect
            password = self.load_password()
            
            # 4. Inicializar Spark
            self.init_spark()
            
            # 5. Crear datos de ejemplo (reemplaza con tu lógica de archivo 7z)
            self.create_sample_data()
            
            # 6. Ejemplo de procesamiento de archivo 7z (descomenta si tienes un archivo)
            # archivo_7z = "/ruta/a/tu/archivo.7z"
            # if os.path.exists(archivo_7z):
            #     self.process_7z_file(archivo_7z, password)
            
            print("✅ Flujo ejecutado exitosamente")
            
        except Exception as e:
            print(f"❌ Error en el flujo: {e}")
            raise
        finally:
            self.cleanup()

if __name__ == "__main__":
    flujo = Flujo7zSecretPrefFixed()
    flujo.run()