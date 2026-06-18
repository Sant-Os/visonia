# Sistema Integral de Videovigilancia Autónoma (SIVA-T)
## Basado en Arquitecturas Transformer y Visión Computacional

---

## 1. Descripción y Problemática

**La Problemática:**
Los sistemas de videovigilancia tradicionales (CCTV) padecen de una falla fundamental: son pasivos. Dependen por completo de que un operador humano esté mirando la pantalla exacta en el segundo exacto en el que ocurre un incidente. Además, los sistemas clásicos de "detección de movimiento" generan excesivos falsos positivos (activados por cambios de luz o sombras), volviéndolos inútiles para predecir intenciones criminales complejas o emergencias en tiempo real.

**La Solución Implementada:**
Hemos desarrollado un **Centro de Mando Biométrico** proactivo. En lugar de analizar píxeles crudos, el sistema extrae el "esqueleto matemático" de las personas y utiliza Inteligencia Artificial Avanzada (Redes Neuronales Transformers) para entender el flujo del tiempo.

---

## 2. Tecnologías, Librerías y Dependencias

El ecosistema está construido sobre **Python 3.12** utilizando librerías Open-Source de vanguardia:

1. **PyTorch (`torch`, `torch.nn`):** Motor principal de tensores y derivadas usado para crear y entrenar el Cerebro Neuronal (Transformer).
2. **YOLOv8 (`ultralytics`):** Modelos preentrenados (`yolov8n-pose.pt` y `yolov8n.pt`).
3. **OpenCV (`cv2`):** Captura de video en tiempo real y renderizado HUD.
4. **DeepFace:** Framework de reconocimiento facial (RetinaFace y VGG-Face).
5. **HuggingFace Transformers:** Modelo AST (`MIT/ast-finetuned-audioset-10-10-0.4593`) para procesar audio.
6. **PyAudio / WebRTCVAD:** Captura acústica desde el micrófono.
7. **Pandas & NumPy:** Estructuración matemática del CSV de entrenamiento.

### Guía de Instalación
```bash
# 1. Clonar el repositorio
git clone https://github.com/Sant-Os/visonia.git
cd visonia

# 2. Instalar el Core de Machine Learning y Visión Computacional
pip install torch torchvision ultralytics opencv-python pandas numpy

# 3. Instalar librerías secundarias (Rostros y Audio)
pip install deepface tf-keras sounddevice transformers pyaudio
```

---

## 3. Extracción de Puntos y el Secreto de las 34 Coordenadas

### ¿De dónde salen las 34 coordenadas?
El modelo YOLO detecta **17 articulaciones humanas** (Orejas, Ojos, Nariz, Hombros, Codos, Muñecas, Caderas, Rodillas, Tobillos).
Como el video es bidimensional, cada punto tiene un valor horizontal (X) y uno vertical (Y).
> **17 articulaciones × 2 ejes (X, Y) = 34 coordenadas exactas.**

### Centralización Matemática (Normalización)
Para que el modelo Transformer no se confunda con la distancia (una persona lejos vs cerca), aplicamos un **Anclaje de Cadera**:

```python
# Extracto real de pose_extractor.py
kpts_all = r.keypoints.xyn.cpu().numpy()

for i, person_id in enumerate(ids_all):
    kpts = kpts_all[i]
    if len(kpts) == 17:
        # Encontrar la cadera (Puntos 11 y 12)
        center_x = (kpts[11][0] + kpts[12][0]) / 2.0
        center_y = (kpts[11][1] + kpts[12][1]) / 2.0
        
        # Restar el centro pélvico a todo el cuerpo
        normalized_kpts = kpts.copy()
        mask = (normalized_kpts[:, 0] > 0) | (normalized_kpts[:, 1] > 0)
        normalized_kpts[mask, 0] -= center_x
        normalized_kpts[mask, 1] -= center_y
```

---

## 4. Recolección de Datos (El proceso de Registro)

¿Cómo aprende la IA qué es un "Accidente" o un "Forcejeo"? Necesita datos. 
Al presionar la **tecla `[R]`**, se lanza la interfaz de recolección (`collect_data.py`). El usuario se graba a sí mismo ejecutando la acción. 

### El código de guardado en CSV
El sistema toma esas 34 coordenadas cuadro por cuadro y las aplana en un array para guardarlas como una fila en el archivo de Excel (`dataset_poses.csv`):

```python
# Extracto real de collect_data.py
if state == "RECORDING":
    people_landmarks = extractor.extract(frame)
    for p_id, info in people_landmarks.items():
        kpts = info['landmarks'] # Matriz 17x2
        
        # Aplanar la matriz [17,2] a una lista plana de 34 valores
        flat_kpts = kpts.flatten().tolist()
        
        # Insertar la clase (Ej. 'forcejeo') al principio de la lista
        row = [classes[current_class_idx]] + flat_kpts
        data_rows.append(row)

# Al finalizar los 10 segundos, guardar en el disco:
columns = ['class'] + [f'coord_{i}' for i in range(34)]
new_df = pd.DataFrame(data_rows, columns=columns)
new_df.to_csv('dataset_poses.csv', mode='a', header=not file_exists, index=False)
```

---

## 5. El Cerebro Neuronal: Entrenamiento y Arquitectura

Tras guardar los datos, si el usuario presiona la **tecla `[T]`**, arranca el entrenamiento profundo en segundo plano (`action_classifier.py`).

### La Arquitectura del Transformer
Usamos `TransformerEncoderLayer` porque posee "Atención Global", logrando correlacionar el movimiento de una mano en el cuadro 1 con la cabeza en el cuadro 30 de forma instantánea.

```python
import torch.nn as nn

class ActionTransformer(nn.Module):
    def __init__(self, input_dim=34, num_classes=6, hidden_dim=64, num_layers=2):
        super().__init__()
        # Inyecta las 34 coordenadas en un hiperespacio de 64 dimensiones
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Capa Transformer: Analiza la relación temporal (nhead=4)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Reducción matemática a las 6 categorías posibles
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x recibe una matriz de: (Batch, 30_cuadros, 34_coordenadas)
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1) # Promedio el entendimiento del segundo entero
        return self.fc(x)
```

### Proceso de Entrenamiento (Backpropagation)
El script agrupa las filas del CSV en **ventanas de 30 frames** (1 segundo de video). 
Calcula el margen de error usando `CrossEntropyLoss` y ajusta su cerebro usando el optimizador `Adam`. Al final de 20 épocas, compila todo el conocimiento en el archivo `action_model.pth`.

---

## 6. Detección de Armas, Personas y Sabotajes

El sistema ejecuta múltiples sub-redes neuronales en paralelo (`ThreadPoolExecutor`) para interceptar amenazas más allá de las posturas físicas.

### Detección de Objetos Peligrosos (`object_detector.py`)
Utilizamos un modelo YOLO pre-entrenado que escanea todo el video buscando armas. Hemos filtrado el gigantesco diccionario de YOLO (COCO dataset) para enfocarnos estricta y únicamente en 6 objetos críticos.

**El código de definición de amenazas:**
```python
# Extracto real de object_detector.py
class DangerousObjectDetector:
    def __init__(self, model_name='yolov8s.pt'):
        self.model = YOLO(model_name)
        
        # Mapeo de Niveles DEFCON a clases del dataset COCO de YOLO
        self.level_1_classes = [67] # Teléfono Celular
        self.level_2_classes = [24] # Mochila (Alerta de Equipaje Abandonado)
        self.level_4_classes = [43, 34, 76, 39] # Cuchillo, Bate de Béisbol, Tijeras, Botella de vidrio

        self.traducciones = {
            "cell phone": "Celular",
            "backpack": "Mochila",
            "knife": "Cuchillo",
            "baseball bat": "Bate",
            "scissors": "Tijeras",
            "bottle": "Botella"
        }
```

**Si se detecta un Cuchillo (Nivel 4)**, la variable interna del sistema se eleva a DEFCON 4 disparando todas las alertas visuales en rojo y enviando mensajes de Telegram al instante.

### Resumen de Categorías y Alarmas Soportadas:
1. **Comportamiento Físico (Transformer):** `Normal`, `Accidente/Caída`, `Acecho`, `Escape`, `Sumisión` y `Forcejeo`.
2. **Armas (YOLO Objetos):** `Cuchillo`, `Bate`, `Botella`.
3. **Terrorismo (YOLO Tracking):** Si el objeto `Mochila` no se mueve un solo milímetro durante más de 15 segundos, dispara una alarma de "Equipaje Abandonado".
4. **Sabotaje Visual:** Ceguera por Linternas/Láser (`brillo promedio > 240`), o Cámara Tapada (`brillo < 15`).
5. **Rostros Ocultos:** Si se detectan hombros y caderas pero el sistema es incapaz de localizar ojos/nariz, deduce que el intruso lleva "Pasamontañas/Casco".
6. **Auditivo:** Identificación de `Sirenas de Policía`, `Ambulancias`, y `Alarmas de Incendios` interceptando el espectrograma del micrófono interno.
