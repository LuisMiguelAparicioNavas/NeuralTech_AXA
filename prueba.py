import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim

# Cargar el archivo CSV
df = pd.read_csv('synthetic_patients_rehab.csv')

# Limpiar los nombres de las columnas, eliminando espacios adicionales
df.columns = df.columns.str.strip()

# Eliminar las columnas de fraude
df = df.drop(columns=['Fraude_Paciente', 'Fraude_Centro', 'Fraude_Total'], errors='ignore')

# Eliminar filas con valores nulos
df = df.dropna()

# Convertir todas las columnas a numéricas
for column in df.columns:
    if df[column].dtype == 'object':  # Si la columna es de tipo objeto (texto)
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))  # Convertir texto a números

        # 🔹 Agregar un print para ver el mapeo de valores
        print(f"Columna: {column}")
        print(dict(zip(le.classes_, le.transform(le.classes_))))  # Muestra cómo se codifican las categorías
        print("-" * 50)  # Separador para mejor lectura

    else:
        df[column] = pd.to_numeric(df[column], errors='coerce')  # Convertir cualquier valor no numérico a NaN si es necesario

# Eliminar la columna 'ID_Paciente'
df = df.drop(columns=['ID_Paciente'], errors='ignore')

# Ver los primeros registros para confirmar las conversiones
print(df.head())

# Seleccionar las características para el clustering (todas las columnas numéricas)
X = df

# Escalar los datos antes de la clusterización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar KMeans para la clusterización
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)  # Asegurar que `n_init` esté definido
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Ver los resultados de la clusterización
print(df[['Diagnóstico', 'Cluster']].head())

# Convertir el DataFrame a un tensor de PyTorch
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# 🔹 **Reducción de la capacidad del Autoencoder**
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        # Capa de codificación con Dropout
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(True),
            nn.Dropout(0.2)  # Dropout para evitar sobreajuste
        )
        # Capa de decodificación
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Definir las dimensiones de entrada y codificación
input_dim = X_tensor.shape[1]
encoding_dim = 10  # Reducido para forzar el modelo a perder información

# Instanciar el modelo
autoencoder = Autoencoder(input_dim, encoding_dim)

# Definir el optimizador y la función de pérdida
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Entrenar el modelo
epochs = 50
batch_size = 64

for epoch in range(epochs):
    for i in range(0, len(X_tensor), batch_size):
        batch = X_tensor[i:i + batch_size]

        # Enviar los datos al modelo
        output = autoencoder(batch)

        # Calcular la pérdida
        loss = criterion(output, batch)

        # Realizar la retropropagación
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

# 🔹 Guardar el modelo entrenado
model_path = "autoencoder_model.pth"
torch.save(autoencoder.state_dict(), model_path)
print(f"Modelo guardado en: {model_path}")

# Realizar la predicción
with torch.no_grad():
    reconstructed_X = autoencoder(X_tensor)

# Calcular el error de reconstrucción
reconstruction_error = torch.mean((X_tensor - reconstructed_X) ** 2, dim=1)

# 🔹 **Ajuste del umbral** (Percentil 90 en lugar de 95)
threshold = np.percentile(reconstruction_error.numpy(), 90)

# 🔹 **Alternativa: Detección basada en desviación estándar**
mean_error = np.mean(reconstruction_error.numpy())
std_error = np.std(reconstruction_error.numpy())
threshold_std = mean_error + (2 * std_error)  # Dos desviaciones estándar

# Detectar outliers con el nuevo umbral
outliers = reconstruction_error.numpy() > threshold
outliers_std = reconstruction_error.numpy() > threshold_std

print(f"Outliers detectados (percentil 90): {np.sum(outliers)}")
print(f"Outliers detectados (2 std dev): {np.sum(outliers_std)}")

# 🔹 Cargar el modelo guardado en otra sesión si es necesario
autoencoder.load_state_dict(torch.load(model_path))
autoencoder.eval()
print("Modelo cargado correctamente")