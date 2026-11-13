# Monitor de Calidad del Aire con Detección de Anomalías y Asistente de Voz

Este proyecto simula un sistema de monitoreo de gases (COV, CO, CO2, PM2.5). Utiliza el algoritmo PCA (Análisis de Componentes Principales) para detectar fallos en los sensores y cuenta con un asistente de Inteligencia Artificial que te responde por voz sobre el histórico de las últimas 24 horas.

## Capturas de pantalla
*(Aquí debes poner la imagen que me mostraste, la de las 4 gráficas. Ver para creer)*

## Requisitos previos (¡MUY IMPORTANTE!)
Antes de empezar, el usuario debe tener instalado:
1. **Python 3.9 o superior** (Descárgalo de python.org y marca la casilla "Add to PATH" al instalar).
2. **Ollama** (Descárgalo de ollama.com y asegúrate de que esté corriendo en tu computadora. El código usa el modelo `gemma:2b`).
3. Un micrófono funcional y altavoces.

## ¿Cómo ejecutar el programa? (Para usuarios sin experiencia)

### Opción Rápida (Windows)
1. Haz clic en el botón verde **"Code"** arriba a la derecha y selecciona **"Download ZIP"**.
2. Descomprime el archivo ZIP.
3. Haz doble clic en el archivo **`iniciar.bat`** (Yo lo he creado para que instale todo automáticamente).
4. ¡Espera a que se abran las gráficas y empiece a hablar!

### Opción Manual (Si la opción rápida falla)
Abre una terminal o consola en la carpeta del proyecto y escribe:
1. `pip install -r requirements.txt`
2. `python tu_script.py`

## ¿Cómo usar el asistente de voz?
Una vez que el programa inicie, el asistente te saludará por voz. Espera a que diga "Escuchando..." y habla claro. Ejemplos de preguntas:
- *"¿Cuál fue el pico más alto de CO2?"*
- *"Dame un resumen de las últimas 24 horas."*
- *"¿A qué hora hubo más PM2.5?"*

Para salir, di: *"Adiós"* o *"Salir"*.

## 🛠️ Solución de problemas
- **No se escucha nada / Da error de audio:** Revisa que tu micrófono esté configurado como predeterminado en Windows.
- **Error con Ollama:** Asegúrate de haber abierto la aplicación de Ollama en tu computadora antes de ejecutar el código.
