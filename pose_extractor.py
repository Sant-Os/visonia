import cv2
from ultralytics import YOLO
import numpy as np

class PoseExtractor:
    def __init__(self):
        """
        Inicializa YOLOv8-Pose, que es más moderno que MediaPipe y compatible
        con cualquier versión de Python, incluyendo la 3.14.
        """
        self.model = YOLO('yolov8n-pose.pt')

    def extract_landmarks(self, frame):
        """
        Extrae las coordenadas clave usando el Procesador (CPU) para aliviar a la tarjeta gráfica.
        Retorna un diccionario de landmarks por persona y el frame con esqueletos.
        """
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, device='cpu')
        
        people_landmarks = {}
        boxes_info = []
        processed_frame = frame.copy()
        
        for r in results:
            processed_frame = r.plot() # Dibuja los esqueletos y las cajas con la confianza
            
            if r.keypoints is not None and len(r.keypoints) > 0 and r.boxes is not None and r.boxes.id is not None:
                # r.keypoints.xyn es de shape (N_persons, 17, 2)
                # r.boxes.id es de shape (N_persons)
                
                kpts_all = r.keypoints.xyn.cpu().numpy()
                ids_all = r.boxes.id.cpu().numpy().astype(int)
                boxes_all = r.boxes.xyxy.cpu().numpy().astype(int)
                
                for i, person_id in enumerate(ids_all):
                    kpts = kpts_all[i] # (17, 2)
                    box = boxes_all[i] # (x1, y1, x2, y2)
                    
                    if len(kpts) == 17:
                        # --- NORMALIZACIÓN ESPACIAL (CENTRALIZACIÓN POR CADERA - V1) ---
                        # Volvemos a la versión ultra-estable que ancla matemáticamente la cadera al centro de la pantalla
                        # Esto ignora las vibraciones del Bounding Box y genera secuencias limpias.
                        if kpts[11][0] != 0 and kpts[12][0] != 0:
                            center_x = (kpts[11][0] + kpts[12][0]) / 2.0
                            center_y = (kpts[11][1] + kpts[12][1]) / 2.0
                        elif kpts[5][0] != 0 and kpts[6][0] != 0:
                            center_x = (kpts[5][0] + kpts[6][0]) / 2.0
                            center_y = (kpts[5][1] + kpts[6][1]) / 2.0
                        else:
                            valid_kpts = kpts[(kpts[:,0] > 0) | (kpts[:,1] > 0)]
                            if len(valid_kpts) > 0:
                                center_x = np.mean(valid_kpts[:, 0])
                                center_y = np.mean(valid_kpts[:, 1])
                            else:
                                continue # Persona inválida
                                
                        # Restar el centro a todos los puntos válidos
                        normalized_kpts = kpts.copy()
                        mask = (normalized_kpts[:, 0] > 0) | (normalized_kpts[:, 1] > 0)
                        normalized_kpts[mask, 0] -= center_x
                        normalized_kpts[mask, 1] -= center_y
                        
                        landmarks_list = normalized_kpts.flatten().tolist()
                        
                        # Heurística Anti-Máscara (Sabotaje Facial Mejorado)
                        # Puntos: 0:Nariz, 1:OjoIzq, 2:OjoDer, 3:OrejaIzq, 4:OrejaDer
                        face_points = kpts[0:5]
                        
                        # Contamos cuántos de los 5 puntos faciales la IA NO puede ver (coordenadas 0,0)
                        missing_face_points = np.sum(np.all(face_points == 0, axis=1))
                        
                        face_visible = True
                        # Si te tapas la cara, los ojos y la nariz desaparecen. 
                        # Si faltan 3 o más de esos 5 puntos, activamos la alarma.
                        if missing_face_points >= 3:
                            face_visible = False
                            
                        people_landmarks[person_id] = {
                            'landmarks': landmarks_list,
                            'face_visible': face_visible
                        }
                        boxes_info.append((person_id, box))
                
        return people_landmarks, boxes_info, processed_frame

    def close(self):
        pass # YOLO no requiere cerrado manual explícito

if __name__ == "__main__":
    # Prueba rápida
    cap = cv2.VideoCapture(0)
    extractor = PoseExtractor()
    print("Iniciando prueba de PoseExtractor (YOLOv8). Presiona 'q' para salir.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        people_landmarks, boxes_info, processed_frame = extractor.extract_landmarks(frame)
        cv2.imshow('Pose Test', processed_frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
