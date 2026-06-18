import cv2
import time
import subprocess
import threading
from collections import deque
import psutil
import concurrent.futures
import os
import datetime
import numpy as np
from pygrabber.dshow_graph import FilterGraph
from face_registry import FaceRegistry
import tkinter as tk
from tkinter import simpledialog

try:
    import pygame
    pygame.mixer.init()
    pygame_available = True
except ImportError:
    pygame_available = False
from pose_extractor import PoseExtractor
from object_detector import DangerousObjectDetector
from action_classifier import ActionPredictor
from telegram_alert import TelegramAlert
from audio_detector import AudioAlertDetector

class SecurityApp:
    def __init__(self):
        # --- Variables del Sistema ---
        self.camera_index = 0
        self.cap = None
        self.is_running = False
        
        # Threading (Hilos)
        self.ai_thread = None
        self.audio_thread = None
        self.capture_thread = None
        self.processed_frame = None
        self.latest_camera_frame = None
        
        # Módulos IA
        self.pose_extractor = PoseExtractor()
        self.object_detector = DangerousObjectDetector()
        self.action_predictor = ActionPredictor('action_model.pth')
        self.telegram = TelegramAlert(bot_token="", chat_id="")
        self.audio_detector = AudioAlertDetector()
        
        # Estado de Audio
        self.audio_alert_detected = False
        self.audio_alert_names = []
        
        self.sequence_length = 10
        self.sequences_by_id = {} # Soporte multi-persona
        
        self.last_alert_time = 0
        self.cooldown_seconds = 5
        
        # Temporizadores de Nivel 2
        self.backpack_timers = {} # {id_caja_aprox: tiempo_inicial}
        self.stranger_timers = {} # {person_id: tiempo_inicial}
        
        # Estado Visual
        self.system_state = "Iniciando..."
        self.current_action = "Nadie detectado"
        self.color_bgr = (0, 200, 0)
        self.defcon_level = 1
        
        self.action_history = [] # Para suavizado temporal de acciones
        
        self.recent_logs = []
        self.last_logged_action = ""
        
        # Ejecución Paralela
        self.ai_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        # Carpetas
        if not os.path.exists("evidencias"):
            os.makedirs("evidencias")
        
        self.sequences_by_id = {} 
        self.stranger_timers = {} 
        self.backpack_timers = {} 
        self.last_logged_action = ""
        self.cooldown_seconds = 5.0 
        self.last_alert_time = 0
        
        self.face_registry = FaceRegistry()
        
        # --- Descubrir Cámaras Reales ---
        self.cameras = self.get_available_cameras()
        if not self.cameras:
            self.cameras = ["0: Default Camera"]
            
        self.log_event("Sistema iniciado. Oprime [Q] para salir.")

    def get_available_cameras(self):
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()
            cam_list = []
            for i, dev in enumerate(devices):
                cam_list.append(f"{i}: {dev}")
            return cam_list
        except Exception as e:
            print("No se pudo obtener el nombre real de las cámaras:", e)
            return ["0: Cámara Frontal", "1: Cámara Externa"]

    def log_event(self, msg):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_str = f"[{timestamp}] {msg}"
        self.recent_logs.append(log_str)
        if len(self.recent_logs) > 30: # Ampliamos capacidad de 10 a 30
            self.recent_logs.pop(0)

    def cycle_camera(self):
        if not self.cameras: return
        self.camera_index = (self.camera_index + 1) % len(self.cameras)
        self.log_event(f"Cambio a cámara: {self.cameras[self.camera_index]}")
        self.start_camera()

    def start_camera(self):
        self.is_running = False
        
        if self.ai_thread and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=1.0)
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
            
        if self.cap is not None:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        
        if self.cap.isOpened():
            self.is_running = True
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            self.capture_thread = threading.Thread(target=self.camera_capture_loop, daemon=True)
            self.capture_thread.start()
            
            self.ai_thread = threading.Thread(target=self.ai_processing_loop, daemon=True)
            self.ai_thread.start()
            
            self.audio_thread = threading.Thread(target=self.audio_processing_loop, daemon=True)
            self.audio_thread.start()
        else:
            self.log_event("[ERROR] Cámara no disponible.")

    def camera_capture_loop(self):
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.latest_camera_frame = frame
            else:
                time.sleep(0.01)

    def audio_processing_loop(self):
        while self.is_running:
            danger, names = self.audio_detector.listen_and_detect(duration_seconds=2)
            self.audio_alert_detected = danger
            if danger:
                self.audio_alert_names = names

    def ai_processing_loop(self):
        while self.is_running:
            frame = self.latest_camera_frame
            if frame is None:
                time.sleep(0.01)
                continue
                
            frame = frame.copy()
                
            future_pose = self.ai_pool.submit(self.pose_extractor.extract_landmarks, frame)
            future_obj = self.ai_pool.submit(self.object_detector.detect, frame)
            
            people_landmarks, boxes_info, processed_frame = future_pose.result()
            danger_objects_info = future_obj.result()
            
            for (x1, y1, x2, y2, name, conf, risk_level) in danger_objects_info:
                if risk_level == 4:
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(processed_frame, f"AMENAZA: {name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                elif risk_level == 2:
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(processed_frame, f"Sospechoso: {name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            current_level = 1
            action_desc = "Normal"
            
            mean_brightness = np.mean(frame)
            if mean_brightness < 15:
                current_level = max(current_level, 3)
                action_desc = "Intento de Sabotaje: El lente de la camara ha sido obstruido."
            elif mean_brightness > 240:
                current_level = max(current_level, 3)
                action_desc = "Intento de Sabotaje: Enceguecimiento por luz/laser detectado."
                
            if self.audio_alert_detected:
                a_name = self.audio_alert_names[0].lower()
                if "siren" in a_name or "alarm" in a_name:
                    current_level = max(current_level, 4)
                    action_desc = f"Alerta auditiva critica: Sirena o alarma ({self.audio_alert_names[0]})"
                else:
                    current_level = max(current_level, 2)
                    action_desc = f"Sonido sospechoso detectado en el perimetro: {self.audio_alert_names[0]}"
            
            current_time = time.time()
            for (x1, y1, x2, y2, name, conf, risk_level) in danger_objects_info:
                if risk_level == 4:
                    current_level = max(current_level, 4)
                    action_desc = f"Peligro: Sujeto detectado empunando un/una {name}."
                elif risk_level == 2 and name == "Mochila":
                    box_id = f"{x1//50}_{y1//50}"
                    if box_id not in self.backpack_timers:
                        self.backpack_timers[box_id] = current_time
                    elif (current_time - self.backpack_timers[box_id]) > 15.0:
                        current_level = max(current_level, 2)
                        action_desc = "Precaucion: Se ha detectado equipaje desatendido por mas de 15s."
                elif risk_level == 1:
                    pass
            
            current_backpack_hashes = [f"{x1//50}_{y1//50}" for (x1, y1, x2, y2, name, conf, rl) in danger_objects_info if name == "Mochila"]
            self.backpack_timers = {k: v for k, v in self.backpack_timers.items() if k in current_backpack_hashes}
            
            current_ids = list(people_landmarks.keys())
            ids_to_remove = [pid for pid in self.sequences_by_id.keys() if pid not in current_ids]
            for pid in ids_to_remove:
                del self.sequences_by_id[pid]
                if pid in self.stranger_timers:
                    del self.stranger_timers[pid]

            for person_id, info in people_landmarks.items():
                landmarks = info['landmarks']
                face_visible = info['face_visible']
                
                if not face_visible:
                    current_level = max(current_level, 4)
                    action_desc = "Alerta: Sujeto sospechoso ocultando su identidad (Mascara/Casco)."
                
                # RECONOCIMIENTO FACIAL ----------------------------------
                identity = "Desconocido"
                box = None
                for bid, b in boxes_info:
                    if bid == person_id:
                        box = b
                        break
                        
                if box is not None:
                    if person_id not in self.face_registry.known_identities:
                        self.face_registry.known_identities[person_id] = "Escaneando..."
                        self.face_registry.identify_person_async(frame, box, person_id)
                    else:
                        identity = self.face_registry.known_identities[person_id]

                if identity != "Desconocido" and identity != "Escaneando...":
                    cv2.putText(processed_frame, f"[Autorizado: {identity}]", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)
                else:
                    if identity == "Escaneando...":
                        cv2.putText(processed_frame, "[Escaneando Rostro...]", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)
                        
                    if person_id not in self.stranger_timers:
                        self.stranger_timers[person_id] = current_time
                    elif (current_time - self.stranger_timers[person_id]) > 15.0:
                        current_level = max(current_level, 2)
                        if current_level == 2:
                            action_desc = "Precaucion: Sujeto no identificado merodeando en el area (>15s)."
                # --------------------------------------------------------
                
                if person_id not in self.sequences_by_id:
                    self.sequences_by_id[person_id] = deque(maxlen=self.sequence_length)
                
                self.sequences_by_id[person_id].append(landmarks)
                
                action = "Analizando..."
                if len(self.sequences_by_id[person_id]) == self.sequence_length:
                    action = self.action_predictor.predict(list(self.sequences_by_id[person_id]))
                
                if action in ['sumision', 'forcejeo']:
                    current_level = max(current_level, 4)
                    action_desc = f"Violencia fisica confirmada: {action.capitalize()}"
                elif action in ['caida', 'accidente']:
                    current_level = max(current_level, 3)
                    action_desc = f"Emergencia medica: Persona ha sufrido un accidente/caida."
                elif action in ['acecho']:
                    current_level = max(current_level, 3)
                    action_desc = f"Comportamiento inusual: Posible acecho o espionaje detectado."
                elif action in ['escape']:
                    current_level = max(current_level, 3)
                    action_desc = f"Comportamiento inusual: Persona huyendo apresuradamente."
                    
                for bid, box in boxes_info:
                    if bid == person_id:
                        x1, y1, x2, y2 = box
                        color_box = (0, 255, 0) if person_id == 1 else (0, 255, 255)
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color_box, 1)
                        if action != "normal":
                            cv2.putText(processed_frame, action.upper(), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        break

            # --- Suavizado Temporal ---
            self.action_history.append((current_level, action_desc))
            if len(self.action_history) > 7:
                self.action_history.pop(0)
                
            from collections import Counter
            counts = Counter([a for l, a in self.action_history])
            smoothed_action = counts.most_common(1)[0][0]
            
            smoothed_level = 1
            for l, a in self.action_history:
                if a == smoothed_action:
                    smoothed_level = max(smoothed_level, l)
            
            current_level = smoothed_level
            action_desc = smoothed_action
            # ---------------------------

            self.defcon_level = current_level
            if current_level == 4:
                self.system_state = "AMENAZA CRITICA"
                self.color_bgr = (0, 0, 255) # Rojo en BGR
                self.current_action = action_desc
                if pygame_available and not pygame.mixer.music.get_busy():
                    pass
            elif current_level == 3:
                self.system_state = "INCIDENTE PROGRESO"
                self.color_bgr = (0, 165, 255) # Naranja en BGR
                self.current_action = action_desc
            elif current_level == 2:
                self.system_state = "ALERTA TEMPRANA"
                self.color_bgr = (0, 255, 255) # Amarillo en BGR
                self.current_action = action_desc
            else:
                self.system_state = "ESTADO NORMAL"
                self.color_bgr = (0, 200, 0) # Verde en BGR
                if len(current_ids) > 0:
                    self.current_action = action_desc if action_desc != "Normal" else f"Perimetro seguro. ({len(current_ids)} personas circulando)"
                else:
                    self.current_action = "Perimetro despejado, nadie detectado."
                
            if current_level >= 3 and (time.time() - self.last_alert_time) > self.cooldown_seconds:
                self.last_alert_time = time.time()
                
            if self.current_action != self.last_logged_action and self.defcon_level > 1:
                log_msg = f"{self.current_action}"
                self.log_event(log_msg)
                
                with open("alertas.log", "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {log_msg}\n")
                    
                filename = f"evidencias/alerta_{datetime.datetime.now().strftime('%H_%M_%S')}.jpg"
                cv2.imwrite(filename, processed_frame)
                
                self.last_logged_action = self.current_action
            elif self.defcon_level == 1:
                self.last_logged_action = ""
            
            self.processed_frame = processed_frame.copy()

    def draw_wrapped_text(self, img, text, pos, font, font_scale, color, thickness, max_width):
        words = text.split(' ')
        lines = []
        current_line = ""
        for word in words:
            test_line = current_line + word + " "
            size, _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            if size[0] > max_width:
                if current_line == "":
                    lines.append(word)
                else:
                    lines.append(current_line.strip())
                    current_line = word + " "
            else:
                current_line = test_line
        if current_line:
            lines.append(current_line.strip())
            
        y = pos[1]
        for line in lines:
            cv2.putText(img, line, (pos[0], y), font, font_scale, color, thickness, cv2.LINE_AA)
            y += int(35 * font_scale) # Line height
        return y

    def render_hud(self):
        if self.processed_frame is None:
            return None
            
        frame = self.processed_frame
        h, w, c = frame.shape
        
        hud_w = 420
        hud = np.ones((h, hud_w, 3), dtype=np.uint8) * 240 # Gris muy claro / Blanco
        
        FONT = cv2.FONT_HERSHEY_DUPLEX
        
        # Título
        cv2.putText(hud, "CENTRO DE MANDO", (20, 35), FONT, 0.7, (50, 50, 50), 2, cv2.LINE_AA)
        cv2.line(hud, (20, 45), (hud_w - 20, 45), (150, 150, 150), 1, cv2.LINE_AA)
        
        # Telemetría (Espaciado reducido)
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        
        cv2.putText(hud, f"Carga CPU: {cpu}%", (20, 75), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(hud, (20, 85), (20 + int(cpu * 3.8), 100), (0, 200, 255), -1)
        cv2.rectangle(hud, (20, 85), (400, 100), (150, 150, 150), 1)
        
        cv2.putText(hud, f"Memoria RAM: {ram}%", (20, 130), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(hud, (20, 140), (20 + int(ram * 3.8), 155), (255, 100, 0), -1)
        cv2.rectangle(hud, (20, 140), (400, 155), (150, 150, 150), 1)
        
        # Estado
        cv2.putText(hud, self.system_state, (20, 200), FONT, 0.6, self.color_bgr, 2, cv2.LINE_AA)
        self.draw_wrapped_text(hud, f"Objetivo: {self.current_action}", (20, 225), FONT, 0.5, (50, 50, 50), 1, 380)
        cv2.rectangle(hud, (20, 250), (400, 265), self.color_bgr, -1)
        
        # Controles
        cv2.putText(hud, "CONTROLES:", (20, 310), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(hud, "[C] Camara   [Q] Salir", (20, 335), FONT, 0.45, (0, 100, 0), 1, cv2.LINE_AA)
        cv2.putText(hud, "[R] Recolectar  [T] Entrenar", (20, 355), FONT, 0.45, (0, 100, 0), 1, cv2.LINE_AA)
        cv2.putText(hud, "[F] Registrar Rostro", (20, 375), FONT, 0.45, (0, 100, 0), 1, cv2.LINE_AA)
        
        # Registros
        cv2.putText(hud, "REGISTRO DE EVENTOS:", (20, 410), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.line(hud, (20, 420), (hud_w - 20, 420), (150, 150, 150), 1, cv2.LINE_AA)
        
        # Dibujar logs con Wrap Text para que no se salgan, empujando hacia abajo
        log_y = 440
        for log in reversed(self.recent_logs): # Mostrar los más recientes arriba para no perderlos abajo
            if log_y > h - 20: # Si se sale de la pantalla, dejamos de dibujar
                break
            log_y = self.draw_wrapped_text(hud, log, (20, log_y), FONT, 0.4, (80, 80, 80), 1, 380)
            
        return np.hstack((frame, hud))

    def check_subprocess(self, proc):
        if proc.poll() is None:
            time.sleep(1)
            self.check_subprocess(proc)
        else:
            self.log_event("Reanudando cámara...")
            self.start_camera()

    def open_collect_data(self):
        self.is_running = False
        if self.ai_thread and self.ai_thread.is_alive(): self.ai_thread.join(timeout=1.0)
        if self.audio_thread and self.audio_thread.is_alive(): self.audio_thread.join(timeout=1.0)
        if self.capture_thread and self.capture_thread.is_alive(): self.capture_thread.join(timeout=1.0)
        if self.cap: self.cap.release()
        cv2.destroyAllWindows()
            
        print("\n[SISTEMA] Abriendo GUI de Recoleccion de Datos...")
        proc = subprocess.Popen(['python', 'collect_data.py', str(self.camera_index)])
        threading.Thread(target=self.check_subprocess, args=(proc,), daemon=True).start()

    def open_training(self):
        self.is_running = False
        if self.ai_thread and self.ai_thread.is_alive(): self.ai_thread.join(timeout=1.0)
        if self.audio_thread and self.audio_thread.is_alive(): self.audio_thread.join(timeout=1.0)
        if self.capture_thread and self.capture_thread.is_alive(): self.capture_thread.join(timeout=1.0)
        if self.cap: self.cap.release()
        cv2.destroyAllWindows()
            
        print("\n[SISTEMA] Entrenando modelo en consola...")
        proc = subprocess.Popen(['cmd.exe', '/c', 'start', '/wait', 'python', 'action_classifier.py', 'train'])
        threading.Thread(target=self.check_subprocess, args=(proc,), daemon=True).start()

    def start(self):
        self.start_camera()
        
        window_name = "Sistema de Seguridad Nativo (Oprime 'q' para salir)"
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        while self.is_running:
            combined = self.render_hud()
            if combined is not None:
                cv2.imshow(window_name, combined)
                
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('c') or key == ord('C'):
                self.cycle_camera()
            elif key == ord('r') or key == ord('R'):
                self.open_collect_data()
            elif key == ord('t') or key == ord('T'):
                self.open_training()
            elif key == ord('f') or key == ord('F'):
                if self.latest_camera_frame is not None:
                    # Mostrar ventana emergente (sobre el fullscreen)
                    root = tk.Tk()
                    root.withdraw() # Ocultar ventana principal
                    root.attributes("-topmost", True) # Asegurar que aparezca encima del fullscreen
                    name = simpledialog.askstring("Registro Facial", "Ingresa tu nombre para autorizarte:", parent=root)
                    root.destroy()
                    
                    if name and name.strip():
                        name = name.strip()
                        self.log_event(f"Registrando rostro para: {name}...")
                        self.face_registry.register_face(self.latest_camera_frame, name)
                    else:
                        self.log_event("Registro cancelado.")
                
        self.on_closing()

    def on_closing(self):
        self.is_running = False
        if self.ai_thread and self.ai_thread.is_alive(): self.ai_thread.join(timeout=1.0)
        if self.audio_thread and self.audio_thread.is_alive(): self.audio_thread.join(timeout=1.0)
        if self.capture_thread and self.capture_thread.is_alive(): self.capture_thread.join(timeout=1.0)
        if self.cap: self.cap.release()
        cv2.destroyAllWindows()
        self.pose_extractor.close()

if __name__ == "__main__":
    app = SecurityApp()
    app.start()
