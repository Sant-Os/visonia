# Sistema Integral de Videovigilancia Autónoma (SIVA-T)
## Basado en Arquitecturas Transformer y Visión Computacional

---

## 1. Fuentes de Datos y Recursos Externos Utilizados

Para la construcción de este sistema, se integraron múltiples modelos de vanguardia pre-entrenados y librerías Open-Source. A continuación, se detallan las fuentes oficiales de donde se extrajeron los recursos:

1. **Ultralytics YOLOv8 (Visión y Postura)**
   * **Repositorio:** [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
   * **Uso:** Extracción de esqueletos biomecánicos (`yolov8n-pose.pt`) y detección de armas/mochilas (`yolov8n.pt`).
2. **DeepFace (Biometría Facial)**
   * **Repositorio:** [https://github.com/serengil/deepface](https://github.com/serengil/deepface)
   * **Uso:** Extracción de rostros mediante *RetinaFace* y reconocimiento mediante el modelo *VGG-Face*.
3. **PyTorch (Motor Neuronal)**
   * **Sitio Web:** [https://pytorch.org/](https://pytorch.org/)
   * **Uso:** Arquitectura base para la red neuronal temporal (`nn.TransformerEncoderLayer`).
4. **HuggingFace Audio Spectrogram Transformer (AST)**
   * **Modelo:** [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
   * **Uso:** Clasificación de frecuencias de audio (Sirenas, Alarmas) en segundo plano.
5. **COCO Dataset (Detección de Objetos)**
   * **Sitio Web:** [https://cocodataset.org/](https://cocodataset.org/)
   * **Uso:** Mapeo de diccionarios (Clase 43 = Cuchillo, Clase 24 = Mochila).

---

## 2. Descripción y Problemática

**La Problemática:** Los sistemas de videovigilancia tradicionales (CCTV) son pasivos y dependen por completo de que un operador humano esté mirando la pantalla. Los sistemas clásicos de "detección de movimiento" generan falsos positivos inútiles.

**La Solución Implementada:** Hemos desarrollado un **Centro de Mando Biométrico**. El sistema extrae el "esqueleto matemático" de las personas y utiliza Inteligencia Artificial Avanzada (Transformers) para entender el flujo del tiempo. Diferencia un abrazo de un estrangulamiento, audita el entorno sonoro, e identifica rostros autorizados.

---

## 3. Implementación Completa: Estructura del Código Fuente

A continuación se detalla cómo está estructurado el código del proyecto y los fragmentos críticos de cada módulo.

### Módulo 1: `pose_extractor.py` (De Píxeles a Matemáticas)
Este archivo se encarga de llamar a YOLOv8, extraer los 17 puntos del cuerpo y aplicar la **Centralización de Cadera** para que la IA entienda posturas sin importar la distancia a la cámara.

```python
from ultralytics import YOLO
import numpy as np

class PoseExtractor:
    def __init__(self, model_name='yolov8n-pose.pt'):
        self.model = YOLO(model_name)
        
    def extract(self, frame):
        # 1. Extraer los puntos clave con YOLO
        results = self.model.track(frame, persist=True, verbose=False)
        r = results[0]
        
        kpts_all = r.keypoints.xyn.cpu().numpy()
        ids_all = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else []
        
        people_data = {}
        for i, person_id in enumerate(ids_all):
            kpts = kpts_all[i]
            if len(kpts) == 17:
                # 2. Localizar la Cadera (Puntos 11 y 12)
                center_x = (kpts[11][0] + kpts[12][0]) / 2.0
                center_y = (kpts[11][1] + kpts[12][1]) / 2.0
                
                # 3. Normalización: Restar la cadera para anclar la persona al origen (0,0)
                normalized_kpts = kpts.copy()
                mask = (normalized_kpts[:, 0] > 0) | (normalized_kpts[:, 1] > 0)
                normalized_kpts[mask, 0] -= center_x
                normalized_kpts[mask, 1] -= center_y
                
                people_data[person_id] = {'landmarks': normalized_kpts}
        return people_data
```

### Módulo 2: `collect_data.py` (Recolección y Guardado en CSV)
Este módulo se lanza al presionar **`[R]`**. Toma las coordenadas normalizadas del `pose_extractor` y las aplana a **34 columnas** para guardarlas en `dataset_poses.csv`.

```python
import pandas as pd
import cv2

# ... (Lógica de GUI)
if state == "RECORDING":
    people_landmarks = extractor.extract(frame)
    for p_id, info in people_landmarks.items():
        kpts = info['landmarks'] # Matriz de 17x2
        
        # Aplanar la matriz a 34 coordenadas lineales
        flat_kpts = kpts.flatten().tolist()
        
        # Añadir la clase (Ej: 'forcejeo') como primera columna
        row = [classes[current_class_idx]] + flat_kpts
        data_rows.append(row)

# Guardar en el disco al finalizar los 10 segundos
columns = ['class'] + [f'coord_{i}' for i in range(34)]
new_df = pd.DataFrame(data_rows, columns=columns)
new_df.to_csv('dataset_poses.csv', mode='a', index=False)
```

### Módulo 3: `action_classifier.py` (La Red Neuronal)
Aquí reside el cerebro del sistema. Se lanza al presionar **`[T]`**. Usa PyTorch para compilar un `Transformer` que analiza 30 cuadros de video continuos.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class ActionTransformer(nn.Module):
    def __init__(self, input_dim=34, num_classes=6, hidden_dim=64, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Transformer con 4 cabezales de atención global
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: (Batch, 30_Frames, 34_Coordenadas)
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1) # Promedio temporal
        return self.fc(x)

# Proceso de Entrenamiento (Backpropagation)
def train():
    # Carga de CSV y Agrupación en Secuencias de 30 Frames
    # ...
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(20):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    torch.save(model.state_dict(), 'action_model.pth')
```

### Módulo 4: `object_detector.py` (Detección de Armas y Objetos)
Extrae amenazas inanimadas. Filtra las 80 clases COCO originales de YOLO a unas cuantas clases de alto riesgo.

```python
from ultralytics import YOLO

class DangerousObjectDetector:
    def __init__(self, model_name='yolov8s.pt'):
        self.model = YOLO(model_name)
        
        # Diccionario COCO Mapeado a Riesgos
        self.level_1_classes = [67] # cell phone
        self.level_2_classes = [24] # backpack (Equipaje abandonado)
        self.level_4_classes = [43, 34, 76, 39] # knife, baseball bat, scissors, bottle

    def detect(self, frame):
        results = self.model(frame, verbose=False, device=0)
        detected_objects = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                
                # Asignación del nivel de peligro
                risk_level = 0
                if cls_id in self.level_4_classes:
                    risk_level = 4 # Amenaza Crítica
                elif cls_id in self.level_2_classes:
                    risk_level = 2 # Precaución
                    
                if risk_level > 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    name = self.model.names[cls_id]
                    detected_objects.append((x1, y1, x2, y2, name, float(box.conf[0]), risk_level))
        return detected_objects
```

### Módulo 5: `app.py` (El Integrador Multihilo)
El archivo principal no procesa IA linealmente. Lanza las inferencias a un `ThreadPoolExecutor` para no bajar de los 30 FPS.

```python
import concurrent.futures
import cv2

class SecurityApp:
    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        # Inicialización de todos los módulos anteriores...
        
    def ai_processing_loop(self):
        while self.running:
            frame = self.latest_camera_frame
            
            # Ejecución en Paralelo (Multihilo)
            future_pose = self.executor.submit(self.pose_extractor.extract, frame)
            future_obj = self.executor.submit(self.object_detector.detect, frame)
            
            # Esperar resultados
            people_landmarks = future_pose.result()
            danger_objects_info = future_obj.result()
            
            # ... Lógica de HUD y Alertas ...
```

---

## 4. Guía de Instalación y Requisitos

Para clonar e instalar este proyecto en una máquina Windows/Linux:

```bash
# 1. Clonar
git clone https://github.com/Sant-Os/visonia.git
cd visonia

# 2. Instalar el Core (PyTorch + YOLO)
pip install torch torchvision ultralytics opencv-python pandas numpy

# 3. Instalar Análisis Secundarios (DeepFace + Audio AST)
pip install deepface tf-keras sounddevice transformers pyaudio
```

Una vez instalado, basta con ejecutar `python app.py` para iniciar el Centro de Mando.
