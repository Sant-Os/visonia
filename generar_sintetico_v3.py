import pandas as pd
import numpy as np

print("Generando datos sinteticos V3 (Centralización por Cadera - V1 Mode)...")

classes = ['normal', 'accidente', 'escape', 'acecho', 'sumision', 'forcejeo']
data = []

# En la V1, las coordenadas (xyn) son relativas a la pantalla (0.0 a 1.0)
# Luego, se les RESTA la cadera. 
# Por lo tanto, la cadera es (0,0). La cabeza está en Y negativo. Los pies en Y positivo.
# Asumimos que la persona ocupa el 80% de la altura de la pantalla (y = -0.4 a +0.4)

def add_noise(kpts, scale=0.015):
    return kpts + np.random.normal(0, scale, kpts.shape)

for cls in classes:
    for seq in range(200): # 200 secuencias de 10 cuadros = 2000 cuadros por clase
        phase = np.random.rand() * 2 * np.pi
        
        for frame_idx in range(10):
            kpts = np.zeros((17, 2))
            t = frame_idx / 10.0
            
            # Postura base (Normal / Parado) centrada en la cadera (0,0)
            # Cabeza 
            kpts[0] = [0.0, -0.4]; kpts[1] = [-0.02, -0.42]; kpts[2] = [0.02, -0.42]
            kpts[3] = [-0.04, -0.4]; kpts[4] = [0.04, -0.4]
            # Hombros
            kpts[5] = [-0.15, -0.3]; kpts[6] = [0.15, -0.3]
            # Codos
            kpts[7] = [-0.18, -0.1]; kpts[8] = [0.18, -0.1]
            # Munecas
            kpts[9] = [-0.2, 0.0]; kpts[10] = [0.2, 0.0]
            # Caderas (Centro = 0,0)
            kpts[11] = [-0.1, 0.0]; kpts[12] = [0.1, 0.0]
            # Rodillas
            kpts[13] = [-0.1, 0.2]; kpts[14] = [0.1, 0.2]
            # Tobillos
            kpts[15] = [-0.1, 0.4]; kpts[16] = [0.1, 0.4]

            if cls == 'normal':
                # Ligero balanceo (incluso al estar parado, uno se mueve un poquito)
                kpts[:, 0] += np.sin(t * np.pi * 2 + phase) * 0.01
                
            elif cls == 'escape':
                # Correr (brazos y piernas alternando)
                swing = np.sin(t * np.pi * 4 + phase)
                kpts[9, 0] += swing * 0.15; kpts[10, 0] -= swing * 0.15 
                kpts[15, 0] -= swing * 0.15; kpts[16, 0] += swing * 0.15 
                kpts[13, 1] -= abs(swing) * 0.1; kpts[14, 1] -= abs(swing) * 0.1 
                
            elif cls == 'accidente':
                # Cayendo (la cabeza y el torso colapsan hacia la cadera y el piso)
                # Al final de la caída, la cabeza está casi al nivel de la cadera (Y=0)
                drop = t * 0.35
                kpts[0:5, 1] += drop # Cabeza baja
                kpts[5:11, 1] += drop * 0.8 # Hombros bajan
                kpts[15, 1] -= drop * 0.5 # Piernas suben un poco
                
            elif cls == 'acecho':
                # Agachado, torso mas cerca de la cadera
                kpts[0:11, 1] += 0.15 
                kpts[13:17, 1] -= 0.05
                # Caminata muy lenta
                swing = np.sin(t * np.pi * 2 + phase) * 0.05
                kpts[15, 0] += swing; kpts[16, 0] -= swing
                
            elif cls == 'sumision':
                # Manos arriba o en la cabeza (relativo a la cadera)
                kpts[9] = [-0.1, -0.5]; kpts[10] = [0.1, -0.5] # Manos muy arriba
                kpts[7] = [-0.2, -0.4]; kpts[8] = [0.2, -0.4] # Codos arriba
                # De rodillas (pies se acercan a la cadera)
                kpts[15, 1] -= 0.2; kpts[16, 1] -= 0.2
                
            elif cls == 'forcejeo':
                # Peleando (manos cerca de la cara, aleatorio rapido)
                kpts[9] = [-0.1 + np.random.normal(0, 0.08), -0.3 + np.random.normal(0, 0.08)]
                kpts[10] = [0.1 + np.random.normal(0, 0.08), -0.3 + np.random.normal(0, 0.08)]
                # Movimiento errático del torso
                kpts[0:11, 0] += np.random.normal(0, 0.05)

            kpts = add_noise(kpts, scale=0.015) # Menos ruido para mayor estabilidad
            row = [cls] + kpts.flatten().tolist()
            data.append(row)

columns = ['class'] + [f'coord_{i}' for i in range(34)]
df = pd.DataFrame(data, columns=columns)
df.to_csv('dataset_poses.csv', index=False)
print(f"Dataset V3 (Estabilidad V1) guardado con {len(df)} cuadros perfectamente centrados en la cadera.")
