import pandas as pd
import numpy as np

def generate_synthetic_data(num_samples_per_class=300):
    print("Generando datos sintéticos (falsos) para probar la arquitectura...")
    
    classes = ['normal', 'caida', 'acecho', 'escape', 'sumision', 'forcejeo']
    num_features = 34 # 17 landmarks * 2 (x, y)
    
    data_rows = []
    
    for cls in classes:
        for _ in range(num_samples_per_class):
            # Base skeleton normalizado (centrado en 0, 0)
            base_pose = np.random.normal(loc=0.0, scale=0.05, size=num_features)
            
            if cls == 'normal':
                # Ligero movimiento, postura erguida
                base_pose[1::2][:10] -= 0.2 
            elif cls == 'caida':
                # Nivel 3: Caída. Cuerpo aplastado horizontalmente
                base_pose[1::2] = np.random.normal(loc=0.0, scale=0.05, size=17) # Poco rango en Y
                base_pose[0::2] = np.random.normal(loc=0.0, scale=0.4, size=17) # Gran rango en X
            elif cls == 'acecho':
                # Nivel 3: Acecho / Agachado. Hombros y caderas muy juntos en Y
                base_pose[1::2] += 0.2
                base_pose[1::2][5:7] += 0.3 # Hombros bajos
                base_pose[1::2][11:13] -= 0.1 # Caderas comprimidas
            elif cls == 'escape':
                # Nivel 3: Paso acelerado. Nodos de pies y rodillas con gran dispersión en X (Zancada)
                base_pose[0::2][13:17] = np.random.normal(loc=0.0, scale=0.5, size=4)
                base_pose[1::2][:10] -= 0.1 # Inclinado hacia adelante
            elif cls == 'sumision':
                # Nivel 4: Manos arriba. Muñecas (9,10) en Y muy negativas (arriba), codos doblados
                base_pose[1::2][9:11] -= 0.6 # Muñecas muy arriba
                base_pose[1::2][7:9] -= 0.3 # Codos arriba
            elif cls == 'forcejeo':
                # Nivel 4: Ruido de alta frecuencia y dispersión total de brazos, asimétrico
                base_pose[10:22] = np.random.normal(loc=0.0, scale=0.4, size=12) # Brazos locos
                base_pose[1::2] += np.random.normal(loc=0.0, scale=0.1, size=17) # Inestabilidad de cuerpo
            
            # Limitar a -1.0 a 1.0 (coordenadas relativas)
            base_pose = np.clip(base_pose, -1.0, 1.0)
            
            row = [cls] + base_pose.tolist()
            data_rows.append(row)
            
    columns = ['class'] + [f'coord_{i}' for i in range(num_features)]
    df = pd.DataFrame(data_rows, columns=columns)
    
    df.to_csv('dataset_poses.csv', index=False)
    print(f"¡Listo! Se creó 'dataset_poses.csv' con {len(df)} ejemplos sintéticos.")

if __name__ == "__main__":
    generate_synthetic_data()
