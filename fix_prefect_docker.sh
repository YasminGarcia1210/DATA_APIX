#!/bin/bash

echo "🔧 Solucionando problema de configuración de Prefect en Docker"
echo "=============================================================="

# Verificar que Prefect esté corriendo en el host
echo "1. Verificando servidor Prefect en el host..."
if ! curl -s http://localhost:4200/api/health > /dev/null; then
    echo "❌ Servidor Prefect no está corriendo en el host"
    echo "   Ejecuta: prefect server start"
    exit 1
fi
echo "✅ Servidor Prefect está corriendo en el host"

# Copiar los archivos corregidos al contenedor
echo ""
echo "2. Copiando archivos corregidos al contenedor Docker..."
docker cp fix_prefect_config.py spark-master:/opt/bitnami/spark/app/TRAZABILIDAD_1/
docker cp flujo_7z_secret_pref_fixed.py spark-master:/opt/bitnami/spark/app/TRAZABILIDAD_1/

echo "✅ Archivos copiados al contenedor"

# Configurar Prefect en el contenedor
echo ""
echo "3. Configurando Prefect en el contenedor Docker..."
docker exec -it spark-master python3 /opt/bitnami/spark/app/TRAZABILIDAD_1/fix_prefect_config.py

# Verificar que el bloque esté accesible
echo ""
echo "4. Verificando acceso al bloque desde el contenedor..."
docker exec -it spark-master prefect block ls

# Ejecutar el trabajo de Spark corregido
echo ""
echo "5. Ejecutando el trabajo de Spark con la configuración corregida..."
docker exec -it spark-master /opt/bitnami/spark/bin/spark-submit \
  --jars /opt/bitnami/spark/jars/iceberg-spark-runtime-3.5_2.12-1.6.1.jar \
  --conf spark.sql.catalog.lake=org.apache.iceberg.spark.SparkCatalog \
  --conf spark.sql.catalog.lake.type=hadoop \
  --conf spark.sql.catalog.lake.warehouse=s3a://warehouse/iceberg/ \
  --conf spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions \
  /opt/bitnami/spark/app/TRAZABILIDAD_1/flujo_7z_secret_pref_fixed.py

echo ""
echo "🎉 ¡Proceso completado!"