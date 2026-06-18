import cv2
import os
import time
from deepface import DeepFace
import threading
import logging

# Desactivar logs pesados de tensorflow
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class FaceRegistry:
    def __init__(self, db_path='conocidos'):
        self.db_path = db_path
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
            
        self.known_identities = {} # person_id -> "Name"
        self.processing = set() # Evitar lanzar multiples hilos para la misma persona
        
    def register_face(self, frame, name="Admin"):
        """ Guarda un cuadro completo en la base de datos con el nombre provisto """
        print(f"\n[INFO] Escaneando rostro para registro: {name}...")
        try:
            timestamp = int(time.time())
            filename = f"{name}_{timestamp}.jpg"
            save_path = os.path.join(self.db_path, filename)
            cv2.imwrite(save_path, frame)
            print(f"[EXITO] Rostro autorizado y guardado como {filename}")
            
            # Borrar cache de deepface para forzar re-lectura en la proxima deteccion
            for file in os.listdir(self.db_path):
                if file.endswith('.pkl'):
                    os.remove(os.path.join(self.db_path, file))
                    
            # Resetear identidades actuales para forzar re-escaneo
            self.known_identities.clear()
            return True
        except Exception as e:
            print(f"[ERROR] No se pudo registrar: {e}")
            return False
            
    def identify_person_async(self, frame, bbox, person_id):
        if person_id in self.processing:
            return
            
        self.processing.add(person_id)
        threading.Thread(target=self._identify_person_task, args=(frame.copy(), bbox, person_id), daemon=True).start()

    def _identify_person_task(self, frame, bbox, person_id):
        x1, y1, x2, y2 = bbox
        
        # Expandir la caja para incluir toda la cabeza y evitar cortes
        h, w = frame.shape[:2]
        padding = 30
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            self.known_identities[person_id] = "Desconocido"
            self.processing.remove(person_id)
            return
            
        temp_path = f"temp_face_{person_id}.jpg"
        cv2.imwrite(temp_path, person_crop)
        
        try:
            # Revisar si existen fotos en la DB
            if not any(f.endswith(('.jpg', '.png')) for f in os.listdir(self.db_path)):
                self.known_identities[person_id] = "Desconocido"
                return
                
            # Buscar en la DB (enforce_detection=False evita crasheos si la imagen es borrosa)
            dfs = DeepFace.find(img_path=temp_path, db_path=self.db_path, enforce_detection=False, silent=True)
            
            if len(dfs) > 0 and len(dfs[0]) > 0:
                matched_path = dfs[0].iloc[0]['identity']
                name = os.path.basename(matched_path).split('_')[0]
                self.known_identities[person_id] = name
            else:
                self.known_identities[person_id] = "Desconocido"
        except Exception as e:
            self.known_identities[person_id] = "Desconocido"
        finally:
            if os.path.exists(temp_path):
                try: os.remove(temp_path)
                except: pass
            if person_id in self.processing:
                self.processing.remove(person_id)
