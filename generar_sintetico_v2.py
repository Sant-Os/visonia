import pandas as pd
import numpy as np

print("Generando datos sinteticos V2 (Alineados al Bounding Box de YOLOv8)...")

classes = ['normal', 'accidente', 'escape', 'acecho', 'sumision', 'forcejeo']
data = []

# YOLOv8 entrega coordenadas normalizadas de 0.0 a 1.0 (relativas a la caja delimitadora de la persona)
def add_noise(kpts, scale=0.03):
    return np.clip(kpts + np.random.normal(0, scale, kpts.shape), 0.0, 1.0)

for cls in classes:
    for seq in range(200): # 200 secuencias de 10 cuadros = 2000 cuadros por clase
        phase = np.random.rand() * 2 * np.pi
        
        for frame_idx in range(10):
            kpts = np.zeros((17, 2))
            t = frame_idx / 10.0
            
            # Postura base (Normal / Parado)
            # Cabeza (0.0 a 0.2 en Y)
            kpts[0] = [0.5, 0.1]; kpts[1] = [0.45, 0.08]; kpts[2] = [0.55, 0.08]
            kpts[3] = [0.4, 0.1]; kpts[4] = [0.6, 0.1]
            # Hombros
            kpts[5] = [0.2, 0.2]; kpts[6] = [0.8, 0.2]
            # Codos
            kpts[7] = [0.2, 0.4]; kpts[8] = [0.8, 0.4]
            # Munecas
            kpts[9] = [0.2, 0.5]; kpts[10] = [0.8, 0.5]
            # Caderas
            kpts[11] = [0.35, 0.5]; kpts[12] = [0.65, 0.5]
            # Rodillas
            kpts[13] = [0.35, 0.75]; kpts[14] = [0.65, 0.75]
            # Tobillos
            kpts[15] = [0.35, 0.95]; kpts[16] = [0.65, 0.95]

            if cls == 'normal':
                # Ligero movimiento natural
                kpts[:, 0] += np.sin(t * np.pi * 2 + phase) * 0.02
                
            elif cls == 'escape':
                # Correr (brazos y piernas alternando)
                swing = np.sin(t * np.pi * 4 + phase)
                kpts[9, 0] += swing * 0.3; kpts[10, 0] -= swing * 0.3 
                kpts[15, 0] -= swing * 0.3; kpts[16, 0] += swing * 0.3 
                kpts[13, 1] -= abs(swing) * 0.2; kpts[14, 1] -= abs(swing) * 0.2 
                
            elif cls == 'accidente':
                # Cayendo (la cabeza y el torso bajan rapidamente)
                drop = t * 0.7
                kpts[0:5, 1] += drop
                kpts[5:11, 1] += drop * 0.8
                kpts[11:13, 1] += drop * 0.4
                
            elif cls == 'acecho':
                # Agachado, torso mas cerca de la cadera
                kpts[0:11, 1] += 0.25 
                kpts[13:17, 1] += 0.05
                # Caminata muy lenta
                swing = np.sin(t * np.pi * 2 + phase) * 0.1
                kpts[15, 0] += swing; kpts[16, 0] -= swing
                
            elif cls == 'sumision':
                # Manos arriba o en la cabeza
                kpts[9] = [0.2, 0.0]; kpts[10] = [0.8, 0.0] 
                kpts[7] = [0.2, 0.1]; kpts[8] = [0.8, 0.1] 
                # De rodillas
                kpts[0:13, 1] += 0.3 
                
            elif cls == 'forcejeo':
                # Peleando (manos cerca de la cara, aleatorio rapido)
                kpts[9] = [0.3 + np.random.normal(0, 0.15), 0.2 + np.random.normal(0, 0.15)]
                kpts[10] = [0.7 + np.random.normal(0, 0.15), 0.2 + np.random.normal(0, 0.15)]
                kpts[15, 0] = 0.1; kpts[16, 0] = 0.9 # piernas abiertas

            kpts = add_noise(kpts, scale=0.03)
            row = [cls] + kpts.flatten().tolist()
            data.append(row)

columns = ['class'] + [f'coord_{i}' for i in range(34)]
df = pd.DataFrame(data, columns=columns)
df.to_csv('dataset_poses.csv', index=False)
print(f"Dataset corregido guardado con {len(df)} cuadros perfectos.")
