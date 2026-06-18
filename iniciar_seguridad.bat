@echo off
color 0A
echo ========================================================
echo        SISTEMA DE SEGURIDAD IA (NVIDIA ACCELERATED)
echo ========================================================
echo Cargando Entorno Virtual Aislado (Python 3.11)...

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] No se encontro el entorno virtual .venv.
    pause
    exit /b
)

call .venv\Scripts\activate
echo Iniciando motores neuronales...
python app.py
pause
