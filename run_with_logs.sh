#!/bin/bash
cd ~/Desktop/tesis/MPU6050_ML/gesture_classifier
source venv/bin/activate

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export PYTHONIOENCODING=utf-8


# Crear carpeta logs si no existe
mkdir -p logs

# Generar nombre de archivo con fecha
LOG_FILE="logs/sesion_$(date +%Y%m%d_%H%M%S).txt"

echo "📝 Guardando log en: $LOG_FILE"
echo ""

# Ejecutar con guardado de log
python -u realtime_classifier.py 2>&1 | tee "$LOG_FILE"

# Al finalizar, mostrar resumen
echo ""
echo "📊 Resumen de la sesión:"
echo "   Archivo: $LOG_FILE"
echo "   Tamaño: $(du -h "$LOG_FILE" | cut -f1)"
echo "   Clasificaciones: $(grep -c "🎯" "$LOG_FILE")"
