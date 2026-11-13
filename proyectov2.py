import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import f
import pandas as pd
from datetime import datetime, timedelta

# Importaciones para el LLM y Voz
import ollama
import sounddevice as sd
from faster_whisper import WhisperModel
import pyttsx3

# ==========================================
# 1. SIMULACIÓN Y ENTRENAMIENTO (TU CÓDIGO ORIGINAL)
# ==========================================
np.random.seed(42)
N_train = 500
base_combustion = np.random.rand(N_train)

X_train_raw = np.column_stack((
    base_combustion * 10 + np.random.normal(0, 0.5, N_train), # COV
    base_combustion * 5 + np.random.normal(0, 0.2, N_train),  # CO
    base_combustion * 20 + np.random.normal(0, 1.0, N_train), # CO2
    base_combustion * 15 + np.random.normal(0, 0.8, N_train)  # PM2.5
))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)

n_components = 2
pca = PCA(n_components=n_components)
pca.fit(X_train_scaled)

P_k = pca.components_.T
lambdas = pca.explained_variance_[:n_components]

M = X_train_raw.shape[0]
UCL_T2 = (n_components * (M - 1) * (M + 1)) / (M * (M - n_components)) * f.ppf(0.95, n_components, M - n_components)

Q_train = np.zeros(M)
for i in range(M):
    x = X_train_scaled[i, :]
    scores = x @ P_k
    error = x - (scores @ P_k.T)
    Q_train[i] = np.sum(error**2)
UCL_Q = np.percentile(Q_train, 95)

# ==========================================
# 2. DATOS DE PRUEBA Y FALLOS (TU CÓDIGO ORIGINAL)
# ==========================================
N_test = 200
base_test = np.random.rand(N_test)

X_test_raw = np.column_stack((
    base_test * 10 + np.random.normal(0, 0.5, N_test),
    base_test * 5 + np.random.normal(0, 0.2, N_test),
    base_test * 20 + np.random.normal(0, 1.0, N_test),
    base_test * 15 + np.random.normal(0, 0.8, N_test)
))

y_verdadero = np.zeros(N_test)
X_test_raw[50:60, 2] += 15.0 
y_verdadero[50:60] = 1 
X_test_raw[120:130, 0] = 5.0 
y_verdadero[120:130] = 1 

X_test_scaled = scaler.transform(X_test_raw)

T2_array = np.zeros(N_test)
Q_array = np.zeros(N_test)
y_predicho = np.zeros(N_test)

for i in range(N_test):
    x = X_test_scaled[i, :]
    scores = x @ P_k
    T2_array[i] = np.sum((scores**2) / lambdas)
    error = x - (scores @ P_k.T)
    Q_array[i] = np.sum(error**2)
    
    if T2_array[i] > UCL_T2 or Q_array[i] > UCL_Q:
        y_predicho[i] = 1

# ==========================================
# 3. MATRIZ DE CONFUSIÓN (TU CÓDIGO ORIGINAL)
# ==========================================
cm = confusion_matrix(y_verdadero, y_predicho)

# ==========================================
# 4. GRÁFICAS (TU CÓDIGO ORIGINAL)
# ==========================================
plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False, 
            xticklabels=['Normal (0)', 'Anomalía (1)'], 
            yticklabels=['Normal (0)', 'Anomalía (1)'])
plt.title('Matriz de Confusión del Sistema PCA')
plt.xlabel('Lo que predijo el sistema')
plt.ylabel('La realidad (Fallos inyectados)')

plt.subplot(2, 2, 2)
plt.plot(y_verdadero, label='Fallo Real (Verdad)', color='black', linewidth=2, linestyle='--')
plt.plot(y_predicho, label='Alarma del Sistema', color='red', alpha=0.6)
plt.yticks([0, 1], ['Normal', 'Alarma'])
plt.title('Comparación: Fallo Real vs Detección')
plt.xlabel('Muestra (Tiempo)')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(T2_array, label="T2 Real", color='blue')
plt.axhline(y=UCL_T2, color='r', linestyle='--', label=f'UCL T2 ({UCL_T2:.2f})')
plt.title("Estadístico T2 (Magnitud)")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(Q_array, label="Q Real", color='green')
plt.axhline(y=UCL_Q, color='r', linestyle='--', label=f'UCL Q ({UCL_Q:.2f})')
plt.title("Estadístico Q (Fallo de Sensor)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("REPORTE DE RENDIMIENTO DEL ALGORITMO")
print("="*50)
print(classification_report(y_verdadero, y_predicho, target_names=['Normal', 'Anomalía']))
print("="*50)


# ==========================================
# 5. SIMULACIÓN DE HISTÓRICO PARA EL LLM
# ==========================================
print("\n🤖 Iniciando Asistente de Voz con IA...")
print("⏳ Cargando modelo de reconocimiento de voz (Whisper)...")

# Creamos una tabla con las últimas 24 horas de datos (Simulando Firebase)
horas = [(datetime.now() - timedelta(hours=i)).strftime("%Y-%m-%d %H:%M") for i in range(24, -1, -1)]
datos_historicos = []
for i, hora in enumerate(horas):
    # Simulamos picos en horas específicas
    pico_co2 = 1200 if "05:00" in hora else np.random.randint(400, 800)
    pico_pm25 = 25 if "18:00" in hora else np.random.randint(5, 15)
    
    datos_historicos.append({
        "Hora": hora,
        "CO2_ppm": pico_co2,
        "PM2.5_ugm3": pico_pm25,
        "COV_ppb": np.random.randint(100, 500),
        "CO_ppm": round(np.random.uniform(0.5, 3.0), 2)
    })

df_hist = pd.DataFrame(datos_historicos)
tabla_texto = df_hist.to_string(index=False) 

# Configurar motor de voz (TTS)
motor_voz = pyttsx3.init()
motor_voz.setProperty('rate', 150) 

def hablar(texto):
    motor_voz.say(texto)
    motor_voz.runAndWait()

# Inicializar Whisper (se descarga la primera vez)
modelo_whisper = WhisperModel("base", device="cpu", compute_type="int8")

def escuchar():
    print("\n🎤 Escuchando... (Habla ahora, tienes 5 segundos)")
    fs = 16000  # Frecuencia de muestreo recomendada para Whisper
    try:
        # Grabar 5 segundos de audio
        grabacion = sd.rec(int(5 * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()  # Esperar a que termine
        print("Procesando audio con Whisper...")
        
        # Transcribir con faster-whisper
        segmentos, info = modelo_whisper.transcribe(grabacion.flatten(), language="es")
        texto = "".join([segmento.text for segmento in segmentos]).strip().lower()
        
        if texto:
            print(f"🗣️ Tú dijiste: '{texto}'")
            return texto
        else:
            print("❌ No se detectó voz.")
            return ""
    except Exception as e:
        print(f"❌ Error de audio: {e}")
        return ""

# ==========================================
# 6. BUCLE PRINCIPAL DEL ASISTENTE (OLLAMA)
# ==========================================
def iniciar_asistente():
    hablar("Hola. Soy tu asistente de calidad del aire. Tienes los datos de las últimas 24 horas. ¿Qué quieres consultar?")
    
    while True:
        comando = escuchar()
        
        if comando == "":
            continue
            
        if "salir" in comando or "adiós" in comando:
            hablar("Apagando el asistente. Hasta luego.")
            break
            
        print("Consultando a Ollama...")
        
        prompt = f"""
        Eres un asistente experto en calidad del aire. 
        Tienes la siguiente tabla con el histórico de datos de las últimas 24 horas de un laboratorio:
        
        {tabla_texto}
        
        El usuario te acaba de preguntar: "{comando}"
        
        Instrucciones:
        1. Responde a la pregunta basándote ESTRICTAMENTE en la tabla de datos proporcionada.
        2. Si el usuario pide un reporte general, resume los promedios o picos más importantes.
        3. Sé conciso, natural y directo. No uses código ni tablas en tu respuesta, solo texto fluido.
        """
        
        try:
            respuesta_llm = ollama.chat(model='gemma:2b', messages=[
                {'role': 'user', 'content': prompt}
            ])
            
            texto_respuesta = respuesta_llm['message']['content']
            
            print("\n" + "="*50)
            print("RESPUESTA DEL ASISTENTE:")
            print(texto_respuesta)
            print("="*50 + "\n")
            
            hablar(texto_respuesta)
            
        except Exception as e:
            print(f"Error con Ollama: {e}")
            hablar("Lo siento, tuve un problema al conectar con la Inteligencia Artificial.")

if __name__ == "__main__":
    iniciar_asistente()