import pandas as pd
import numpy as np
import os

print("Generando datos biomecánicos sintéticos...")

classes = ['normal', 'accidente', 'escape', 'acecho', 'sumision', 'forcejeo']
data = []

# Esqueleto base (17 puntos). Normalizado y centrado.
base_kpts = np.zeros((17, 2))
base_kpts[0] = [0, -0.8] # nariz
base_kpts[1] = [-0.1, -0.85]; base_kpts[2] = [0.1, -0.85] # ojos
base_kpts[3] = [-0.2, -0.8]; base_kpts[4] = [0.2, -0.8] # orejas
base_kpts[5] = [-0.3, -0.6]; base_kpts[6] = [0.3, -0.6] # hombros
base_kpts[7] = [-0.4, -0.3]; base_kpts[8] = [0.4, -0.3] # codos
base_kpts[9] = [-0.4, 0.0]; base_kpts[10] = [0.4, 0.0] # munecas
base_kpts[11] = [-0.2, 0.0]; base_kpts[12] = [0.2, 0.0] # caderas
base_kpts[13] = [-0.2, 0.5]; base_kpts[14] = [0.2, 0.5] # rodillas
base_kpts[15] = [-0.2, 0.9]; base_kpts[16] = [0.2, 0.9] # tobillos

def add_noise(kpts, scale=0.02):
    return kpts + np.random.normal(0, scale, kpts.shape)

for cls in classes:
    # Generamos 200 secuencias de 10 cuadros = 2000 cuadros por clase
    for seq in range(200):
        phase = np.random.rand() * 2 * np.pi
        
        for frame_idx in range(10):
            kpts = base_kpts.copy()
            t = frame_idx / 10.0
            
            if cls == 'normal':
                # Caminado ligero o estar de pie
                kpts[13:17, 0] += np.sin(t * np.pi * 2 + phase) * 0.1 
                kpts[7:11, 0] += np.cos(t * np.pi * 2 + phase) * 0.1
                
            elif cls == 'escape':
                # Correr rápido (pasos y braceo largo)
                kpts[13:17, 0] += np.sin(t * np.pi * 4 + phase) * 0.5
                kpts[13:17, 1] -= np.abs(np.cos(t * np.pi * 4 + phase)) * 0.3
                kpts[7:11, 0] += np.cos(t * np.pi * 4 + phase) * 0.5
                kpts[7:11, 1] -= np.abs(np.sin(t * np.pi * 4 + phase)) * 0.3
                
            elif cls == 'accidente':
                # Caída libre hacia abajo
                fall_offset = t * 1.8 
                kpts[:, 1] += fall_offset
                kpts[7:11, 1] -= fall_offset * 0.5 # brazos intentando protegerse
                
            elif cls == 'acecho':
                # Agachado, pasos cortos
                kpts[:, 1] += 0.4 # cuerpo mas abajo
                kpts[13:17, 1] -= 0.15 # rodillas flexionadas
                kpts[:, 0] += t * 0.05 # avance muy lento
                
            elif cls == 'sumision':
                # Manos arriba o en la nuca, cuerpo rindiéndose
                kpts[9:11, 1] = -1.0 # munecas arriba de la cabeza
                kpts[7:9, 1] = -0.8  # codos arriba
                kpts[:, 1] += 0.2    # ligeramente agachado
                
            elif cls == 'forcejeo':
                # Peleando, brazos caóticos, posturas aleatorias rápidas
                kpts[7:11, 0] += np.random.normal(0, 0.4, (4,))
                kpts[7:11, 1] += np.random.normal(0, 0.4, (4,))
                kpts[13:17, 0] += np.random.normal(0, 0.2, (4,))
                
            kpts = add_noise(kpts, scale=0.03)
            row = [cls] + kpts.flatten().tolist()
            data.append(row)

columns = ['class'] + [f'coord_{i}' for i in range(34)]
df = pd.DataFrame(data, columns=columns)

if os.path.exists('dataset_poses.csv'):
    # Hacemos un backup del original por si acaso
    if os.path.exists('dataset_poses_backup.csv'):
         os.remove('dataset_poses_backup.csv')
    os.rename('dataset_poses.csv', 'dataset_poses_backup.csv')

df.to_csv('dataset_poses.csv', index=False)
print(f"¡EXITO! Creado dataset_poses.csv con {len(df)} cuadros perfectamente categorizados.")
