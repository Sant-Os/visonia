import cv2
import pandas as pd
import time
import os
import sys
import numpy as np
from pose_extractor import PoseExtractor

def collect_data_gui(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    extractor = PoseExtractor()
    
    classes = ['normal', 'accidente', 'escape', 'acecho', 'sumision', 'forcejeo']
    current_class_idx = 0
    
    state = "IDLE" # IDLE, COUNTDOWN, RECORDING
    countdown_start = 0
    record_start = 0
    duration_seconds = 10
    fps_target = 10
    
    data_rows = []
    
    cv2.namedWindow('Recolector de Datos', cv2.WINDOW_NORMAL)
    
    print(f"--- Recolector GUI Iniciado en Camara {camera_index} ---")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        h, w, c = frame.shape
        # Panel lateral
        panel_w = 300
        panel = np.ones((h, panel_w, 3), dtype=np.uint8) * 40
        
        FONT = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(panel, "RECOLECTOR DE DATOS", (20, 30), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(panel, (20, 40), (panel_w-20, 40), (100, 100, 100), 1)
        
        cv2.putText(panel, "Instrucciones:", (20, 70), FONT, 0.5, (200, 200, 200), 1)
        cv2.putText(panel, "[W]/[S] Cambiar Accion", (20, 100), FONT, 0.5, (100, 255, 100), 1)
        cv2.putText(panel, "[SPACE] Grabar (10s)", (20, 130), FONT, 0.5, (100, 255, 100), 1)
        cv2.putText(panel, "[Q] Volver al Sistema", (20, 160), FONT, 0.5, (100, 100, 255), 1)
        
        cv2.putText(panel, "Accion a Grabar:", (20, 220), FONT, 0.5, (255, 255, 255), 1)
        
        # Lista de clases
        y = 250
        for i, cls_name in enumerate(classes):
            color = (0, 255, 255) if i == current_class_idx else (100, 100, 100)
            prefix = ">> " if i == current_class_idx else "   "
            cv2.putText(panel, f"{prefix}{cls_name.upper()}", (20, y), FONT, 0.6, color, 1)
            y += 30
            
        processed_frame = frame.copy()
        
        if state == "IDLE":
            cv2.putText(panel, "LISTO PARA GRABAR", (20, h - 30), FONT, 0.6, (0, 255, 0), 1)
            
        elif state == "COUNTDOWN":
            elapsed = time.time() - countdown_start
            remains = 3 - int(elapsed)
            if remains <= 0:
                state = "RECORDING"
                record_start = time.time()
                data_rows = []
            else:
                cv2.putText(panel, f"PREPARATE: {remains}", (20, h - 30), FONT, 0.7, (0, 165, 255), 2)
                cv2.putText(processed_frame, str(remains), (w//2-50, h//2), FONT, 5, (0, 165, 255), 5)
                
        elif state == "RECORDING":
            elapsed = time.time() - record_start
            remains = duration_seconds - int(elapsed)
            if remains <= 0:
                state = "IDLE"
                # Guardar CSV
                columns = ['class'] + [f'coord_{i}' for i in range(34)]
                df = pd.DataFrame(data_rows, columns=columns)
                filename = 'dataset_poses.csv'
                if os.path.exists(filename):
                    df.to_csv(filename, mode='a', header=False, index=False)
                else:
                    df.to_csv(filename, index=False)
                print(f"[EXITO] Guardados {len(df)} cuadros para la clase {classes[current_class_idx]}")
            else:
                cv2.putText(panel, f"GRABANDO: {remains}s", (20, h - 30), FONT, 0.7, (0, 0, 255), 2)
                
                people_landmarks, boxes_info, processed_frame = extractor.extract_landmarks(frame)
                if len(people_landmarks) > 0:
                    first_person_id = list(people_landmarks.keys())[0]
                    landmarks = people_landmarks[first_person_id]['landmarks']
                    if len(landmarks) == 34:
                        row = [classes[current_class_idx]] + landmarks
                        data_rows.append(row)
                        
                cv2.circle(processed_frame, (30, 30), 15, (0, 0, 255), -1)
                
        combined = np.hstack((processed_frame, panel))
        cv2.imshow('Recolector de Datos', combined)
        
        key = cv2.waitKey(int(1000/fps_target)) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('w') or key == ord('W'):
            current_class_idx = (current_class_idx - 1) % len(classes)
        elif key == ord('s') or key == ord('S'):
            current_class_idx = (current_class_idx + 1) % len(classes)
        elif key == 32: # SPACE
            if state == "IDLE":
                state = "COUNTDOWN"
                countdown_start = time.time()
                
    cap.release()
    cv2.destroyAllWindows()
    extractor.close()

if __name__ == "__main__":
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    collect_data_gui(cam_idx)
