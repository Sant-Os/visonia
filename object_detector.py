import cv2
from ultralytics import YOLO

class DangerousObjectDetector:
    def __init__(self, model_name='yolov8s.pt'):
        """
        Inicializa el detector con YOLOv8 Small (yolov8s.pt).
        """
        self.model = YOLO(model_name)
        
        # Mapeo de Niveles DEFCON a clases COCO
        self.level_1_classes = [67] # cell phone
        self.level_2_classes = [24] # backpack
        self.level_4_classes = [43, 34, 76, 39] # knife, baseball bat, scissors, bottle

        # Diccionario de Traducción al Español
        self.traducciones = {
            "cell phone": "Celular",
            "backpack": "Mochila",
            "knife": "Cuchillo",
            "baseball bat": "Bate",
            "scissors": "Tijeras",
            "bottle": "Botella"
        }

    def detect(self, frame):
        """
        Detecta objetos en el frame usando la NVIDIA GPU (device=0).
        Retorna la lista de objetos: [(x1, y1, x2, y2, name, conf, risk_level), ...]
        """
        results = self.model(frame, verbose=False, device=0)
        
        detected_objects = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                
                risk_level = 0
                if cls_id in self.level_1_classes:
                    risk_level = 1
                elif cls_id in self.level_2_classes:
                    risk_level = 2
                elif cls_id in self.level_4_classes:
                    risk_level = 4
                    
                if risk_level > 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    name_eng = self.model.names[cls_id]
                    name_esp = self.traducciones.get(name_eng, name_eng) # Traducir si existe
                    detected_objects.append((x1, y1, x2, y2, name_esp, conf, risk_level))
                    
        return detected_objects

if __name__ == "__main__":
    # Prueba rápida
    cap = cv2.VideoCapture(0)
    detector = DangerousObjectDetector()
    print("Iniciando prueba de ObjectDetector. Busca un cuchillo o tijeras. Presiona 'q' para salir.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        danger, names, processed_frame = detector.detect(frame)
        
        if danger:
            print(f"¡PELIGRO DETECTADO!: {names}")
            
        cv2.imshow('Object Test', processed_frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
