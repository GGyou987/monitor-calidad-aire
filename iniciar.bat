@echo off
echo Instalando los componentes necesarios. Por favor espera...
pip install -r requirements.txt
echo.
echo Iniciando el sistema de calidad del aire...
python monitor_aire.py
pause
