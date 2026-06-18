import sounddevice as sd
import numpy as np
import time
from transformers import pipeline
import logging
import os

# Usar un servidor espejo (mirror) para saltar bloqueos regionales de HuggingFace
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Silenciar las advertencias inofensivas de HuggingFace sobre pipelines
logging.getLogger("transformers").setLevel(logging.ERROR)

class AudioAlertDetector:
    def __init__(self):
        """
        Inicializa el clasificador de audio basado en el modelo AST entrenado con AudioSet.
        Descargará ~340MB la primera vez que se ejecute.
        """
        print("[AUDIO] Cargando modelo de Inteligencia Artificial para audio (HuggingFace AST)...")
        try:
            self.classifier = pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593")
            print("[AUDIO] Modelo cargado con éxito. Oído biónico activado.")
            self.is_active = True
        except Exception as e:
            print(f"\n[AUDIO] ERROR REAL: {e}")
            print(f"[AUDIO] ERROR: Tu antivirus o conexión a Internet está bloqueando la descarga desde HuggingFace.")
            print("[AUDIO] El sistema funcionará normalmente, pero la detección de sirenas estará desactivada.\n")
            self.classifier = None
            self.is_active = False
        
        # Clases de AudioSet relevantes para alertas de seguridad
        self.alert_classes = [
            "Siren", 
            "Alarm", 
            "Fire alarm", 
            "Police car (siren)", 
            "Ambulance (siren)", 
            "Smoke detector, smoke alarm", 
            "Emergency vehicle",
            "Burglar alarm"
        ]
        
    def listen_and_detect(self, duration_seconds=2, sample_rate=16000):
        """
        Graba audio del micrófono predeterminado y lo clasifica.
        Retorna (booleano_peligro, lista_nombres_peligro)
        """
        if getattr(self, 'is_active', False) == False:
            time.sleep(2) # Simular tiempo de escucha para no reventar el CPU
            return False, []
            
        try:
            # Grabar audio de forma síncrona (esto corre en su propio hilo en la app principal)
            recording = sd.rec(int(duration_seconds * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
            sd.wait() # Esperar a que termine la grabación
            
            # Aplanar el arreglo de audio a 1D
            audio_data = recording.flatten()
            
            # El modelo AST requiere que el audio esté normalizado
            # Pipeline de HF maneja el preprocesamiento, pasamos el raw numpy array directamente.
            results = self.classifier(audio_data)
            
            # Diccionario de traducción
            traducciones_audio = {
                "Siren": "Sirena",
                "Alarm": "Alarma General",
                "Fire alarm": "Alarma de Incendios",
                "Police car (siren)": "Sirena de Policía",
                "Ambulance (siren)": "Sirena de Ambulancia",
                "Smoke detector, smoke alarm": "Detector de Humo",
                "Emergency vehicle": "Vehículo de Emergencia",
                "Burglar alarm": "Alarma Antirrobos"
            }
            
            danger_detected = False
            danger_names = []
            
            # Analizar los resultados (pipeline devuelve top 5 por defecto)
            for res in results:
                # Si es una alerta conocida y la certeza es mayor al 20% (ajustable)
                if res['label'] in self.alert_classes and res['score'] > 0.20:
                    danger_detected = True
                    nombre_esp = traducciones_audio.get(res['label'], res['label'])
                    danger_names.append(nombre_esp)
                    
            return danger_detected, danger_names
            
        except Exception as e:
            print(f"[AUDIO ERROR] No se pudo capturar o analizar el audio: {e}")
            return False, []

if __name__ == "__main__":
    # Prueba rápida del micrófono
    print("Iniciando prueba de AudioDetector...")
    detector = AudioAlertDetector()
    print("Escuchando por 5 segundos... (haz ruido de sirena si puedes)")
    peligro, nombres = detector.listen_and_detect(duration_seconds=5)
    print(f"Peligro detectado: {peligro}, Tipos: {nombres}")
