# Solución: Error de Configuración de Prefect en Docker

## 🚨 Problema Identificado

El error que estás experimentando ocurre porque el contenedor Docker de Spark está intentando conectarse a un servidor Prefect ephemeral (`http://ephemeral-prefect`) en lugar del servidor Prefect local donde creaste el bloque `photos-7z-password`.

### Error Principal:
```
prefect.exceptions.ObjectNotFound
ValueError: Unable to find block document named photos-7z-password for block type secret
```

### Causa Raíz:
- El bloque `photos-7z-password` fue creado en el servidor Prefect del host (localhost:4200)
- El contenedor Docker está configurado para usar un servidor Prefect diferente
- No hay comunicación entre el contenedor y el servidor Prefect del host

## 🔧 Solución Completa

### Archivos de Solución Creados:

1. **`fix_prefect_config.py`** - Configura la conexión a Prefect en el contenedor
2. **`flujo_7z_secret_pref_fixed.py`** - Versión corregida del trabajo de Spark
3. **`fix_prefect_docker.sh`** - Script automatizado para aplicar la solución

### Paso a Paso Manual:

#### 1. Asegurar que Prefect Server esté corriendo en el host
```bash
# En el host (fuera del contenedor)
prefect server start
```

#### 2. Copiar archivos corregidos al contenedor
```bash
# Copiar el configurador de Prefect
docker cp fix_prefect_config.py spark-master:/opt/bitnami/spark/app/TRAZABILIDAD_1/

# Copiar el trabajo de Spark corregido
docker cp flujo_7z_secret_pref_fixed.py spark-master:/opt/bitnami/spark/app/TRAZABILIDAD_1/
```

#### 3. Configurar Prefect en el contenedor
```bash
# Ejecutar el configurador dentro del contenedor
docker exec -it spark-master python3 /opt/bitnami/spark/app/TRAZABILIDAD_1/fix_prefect_config.py
```

#### 4. Verificar la conexión
```bash
# Verificar que el bloque sea accesible desde el contenedor
docker exec -it spark-master prefect block ls
```

#### 5. Ejecutar el trabajo corregido
```bash
docker exec -it spark-master /opt/bitnami/spark/bin/spark-submit \
  --jars /opt/bitnami/spark/jars/iceberg-spark-runtime-3.5_2.12-1.6.1.jar \
  --conf spark.sql.catalog.lake=org.apache.iceberg.spark.SparkCatalog \
  --conf spark.sql.catalog.lake.type=hadoop \
  --conf spark.sql.catalog.lake.warehouse=s3a://warehouse/iceberg/ \
  --conf spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions \
  /opt/bitnami/spark/app/TRAZABILIDAD_1/flujo_7z_secret_pref_fixed.py
```

### Solución Automatizada:
```bash
# Ejecutar el script de solución completa
chmod +x fix_prefect_docker.sh
./fix_prefect_docker.sh
```

## 🔍 Explicación Técnica

### ¿Qué hace la configuración corregida?

1. **Detección de IP del Host**: El script detecta la IP del host Docker desde dentro del contenedor
2. **Configuración de API URL**: Establece `PREFECT_API_URL` para apuntar al servidor del host
3. **Verificación de Conexión**: Prueba la conectividad antes de ejecutar el trabajo
4. **Manejo de Errores**: Proporciona diagnósticos claros si algo falla

### Código clave en `flujo_7z_secret_pref_fixed.py`:

```python
def configure_prefect_connection(self):
    """Configura la conexión a Prefect en el contenedor Docker"""
    # Obtener IP del host Docker
    result = subprocess.run(['ip', 'route', 'show', 'default'], 
                          capture_output=True, text=True)
    gateway_ip = result.stdout.split()[2]
    
    # Configurar URL de API
    api_url = f"http://{gateway_ip}:4200/api"
    os.environ['PREFECT_API_URL'] = api_url
    
    # Aplicar configuración
    subprocess.run(['prefect', 'config', 'set', f'PREFECT_API_URL={api_url}'])
```

## ✅ Verificación de la Solución

Después de aplicar la solución, deberías ver:

1. **Configuración exitosa**: Mensajes de confirmación de configuración de Prefect
2. **Listado de bloques**: El bloque `photos-7z-password` aparece en `prefect block ls`
3. **Ejecución sin errores**: El trabajo de Spark se ejecuta sin errores de conexión
4. **Contraseña cargada**: Mensajes confirmando que la contraseña se cargó correctamente

## 🚨 Troubleshooting

### Si sigue fallando:

1. **Verificar firewall**: Asegúrate de que el puerto 4200 esté accesible
2. **Verificar red Docker**: Confirma que el contenedor puede alcanzar el host
3. **Verificar servidor Prefect**: Confirma que `prefect server start` esté corriendo
4. **Logs detallados**: Usa los mensajes de debug en los scripts para diagnosticar

### Comandos de diagnóstico:
```bash
# Verificar conectividad desde el contenedor
docker exec -it spark-master curl -v http://172.17.0.1:4200/api/health

# Verificar configuración de Prefect en el contenedor
docker exec -it spark-master prefect config view

# Verificar bloques disponibles
docker exec -it spark-master prefect block ls
```

## 📝 Notas Importantes

- Esta solución asume que el servidor Prefect está corriendo en el puerto 4200 del host
- La IP del gateway Docker puede variar según la configuración de red
- Si usas Docker Compose, puede que necesites configuraciones de red adicionales
- Para ambientes de producción, considera usar variables de entorno o archivos de configuración