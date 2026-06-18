import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

# --- Arquitectura del Transformer ---
class ActionTransformer(nn.Module):
    def __init__(self, input_dim=34, num_classes=4, d_model=64, nhead=4, num_layers=2):
        super(ActionTransformer, self).__init__()
        # Proyectar la entrada a la dimensión del transformer
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Capa Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Clasificador final
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x shape: (Batch, Sequence_Length, Input_Dim)
        x = self.embedding(x)
        
        # Pasar por el transformer
        out = self.transformer_encoder(x)
        
        # Tomar solo el último elemento de la secuencia para clasificar
        out = out[:, -1, :]
        
        # Clasificar
        out = self.fc(out)
        return out

# --- Dataset para PyTorch ---
class PoseDataset(Dataset):
    def __init__(self, csv_file, seq_length=10):
        self.df = pd.read_csv(csv_file)
        self.seq_length = seq_length
        
        # Crear un mapeo automático de texto a número (ej. 'normal' -> 0)
        self.classes = sorted(self.df['class'].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Agrupar secuencias
        self.sequences = []
        self.labels = []
        
        for cls_name in self.classes:
            class_data = self.df[self.df['class'] == cls_name].drop('class', axis=1).values
            # Crear ventanas deslizantes
            for i in range(len(class_data) - seq_length):
                seq = class_data[i : i + seq_length]
                self.sequences.append(seq)
                self.labels.append(self.class_to_idx[cls_name])
                
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return seq, label

# --- Funciones de Entrenamiento y Predicción ---
def train_transformer(csv_file='dataset_poses.csv', model_save_path='action_model.pth', epochs=20):
    if not os.path.exists(csv_file):
        print("ERROR: No existe dataset_poses.csv. Ejecuta collect_data.py primero.")
        return
        
    dataset = PoseDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    num_classes = len(dataset.classes)
    
    model = ActionTransformer(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Iniciando entrenamiento para {num_classes} clases: {dataset.classes}")
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for seqs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")
        
    # Guardar modelo y mapeo de clases
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': dataset.classes
    }, model_save_path)
    print("¡Modelo entrenado y guardado!")

class ActionPredictor:
    def __init__(self, model_path='action_model.pth'):
        self.model = None
        self.classes = []
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            self.classes = checkpoint['classes']
            self.model = ActionTransformer(num_classes=len(self.classes))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print("Cerebro Transformer cargado correctamente.")
        else:
            print("ADVERTENCIA: No se encontró el modelo. Se debe entrenar primero.")

    def predict(self, sequence):
        """ Recibe una lista de 10 cuadros de landmarks y devuelve la clase predicha """
        if self.model is None or len(sequence) < 10:
            return "Desconocido"
            
        with torch.no_grad():
            # Convertir a tensor shape (1, 10, 132)
            seq_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
            output = self.model(seq_tensor)
            
            # Obtener la clase con mayor probabilidad
            _, predicted_idx = torch.max(output.data, 1)
            return self.classes[predicted_idx.item()]

if __name__ == "__main__":
    import sys
    print("--- Módulo Transformer ---")
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train_transformer()
        print("\n[!] Entrenamiento finalizado. Puedes cerrar esta ventana.")
        input("Presiona ENTER para volver al Centro de Mando...")
    else:
        opcion = input("Escribe 'train' para entrenar con tu dataset, o 'test' para salir: ")
        if opcion.lower() == 'train':
            train_transformer()
