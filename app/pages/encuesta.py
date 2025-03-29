import streamlit as st
import pandas as pd
import os

def guardar_respuesta(datos):
    archivo = "respuestas_encuesta.csv"
    nuevo_dato = pd.DataFrame([datos])
    
    if os.path.exists(archivo):
        df = pd.read_csv(archivo)
        df = pd.concat([df, nuevo_dato], ignore_index=True)
    else:
        df = nuevo_dato
    
    df.to_csv(archivo, index=False)
    st.success("¡Encuesta guardada con éxito!")

st.title("Encuesta Médica")

# Opciones predefinidas
centros_medicos = ["Centro A", "Centro B", "Centro C", "Centro D", "Centro E"]
diagnosticos = {
    "Lumbalgia": ["Dolor lumbar", "Rigidez en la espalda", "Espasmos musculares"],
    "Cervicalgia": ["Dolor en cuello", "Rigidez en cuello", "Tensión en hombros"],
    "Síndrome del túnel carpiano": ["Dolor en muñeca", "Hormigueo en dedos", "Debilidad en la mano"],
    "Tendinitis": ["Dolor en codo", "Inflamación en tendón", "Dolor al mover brazo"],
    "Artritis reumatoide": ["Dolor en articulaciones", "Inflamación articular", "Rigidez matutina"],
    "Fibromialgia": ["Dolor generalizado", "Fatiga", "Puntos sensibles"],
    "Hernia discal": ["Dolor irradiado", "Debilidad en piernas", "Entumecimiento"],
    "Bursitis": ["Dolor en articulación", "Inflamación", "Dolor al mover"],
    "Estenosis espinal": ["Dolor en espalda baja", "Debilidad en piernas", "Dificultad para caminar"],
    "Osteoartritis": ["Dolor en articulaciones", "Rigidez", "Inflamación leve"]
}
generos = ["Masculino", "Femenino", "No binario"]
historial_medico = ["Sin antecedentes", "Cirugía previa", "Uso crónico de analgésicos", "Historial de traumatismos", "Fisioterapia previa", "Antecedentes familiares"]
motivos_consulta = ["Dolor persistente", "Dificultad para trabajar", "Recomendado por médico", "Evaluación de síntomas", "Consulta de rutina", "Empeoramiento reciente"]
tratamientos = ["Fisioterapia", "Medicamentos", "Cirugía", "Rehabilitación", "Ejercicios", "Terapia ocupacional"]
actividad_laboral = ["Oficinista", "Obrero", "Ama de casa", "Deportista", "Jubilado", "Estudiante"]

diagnostico = st.selectbox("Diagnóstico", list(diagnosticos.keys()))
sintomas = st.selectbox("Síntomas", diagnosticos[diagnostico])

datos_encuesta = {
    "Centro_Medico": st.selectbox("Centro Médico", centros_medicos),
    "Diagnóstico": diagnostico,
    "Edad": st.number_input("Edad", min_value=0, max_value=120, step=1),
    "Duración_Enfermedad": st.number_input("Duración de la enfermedad (meses)", min_value=0, max_value=120, step=1),
    "Síntomas": sintomas,
    "Género": st.selectbox("Género", generos),
    "Historial_Médico": st.selectbox("Historial Médico", historial_medico),
    "Motivo_Consulta": st.selectbox("Motivo de Consulta", motivos_consulta),
    "Tratamiento_Recomendado": st.selectbox("Tratamiento Recomendado", tratamientos),
    "Actividad_Laboral": st.selectbox("Actividad Laboral", actividad_laboral),
    "Gravedad": st.selectbox("Gravedad", ["Leve", "Moderada", "Grave"]),
    "Sesiones_Necesarias": st.number_input("Sesiones Necesarias", min_value=1, max_value=30, step=1),
    "Sesiones_Solicitadas": st.number_input("Sesiones Solicitadas", min_value=1, max_value=30, step=1),
}

datos_encuesta["Discrepancia_Sesiones"] = datos_encuesta["Sesiones_Solicitadas"] - datos_encuesta["Sesiones_Necesarias"]

import os
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn

# Directorio donde están los archivos guardados
output_dir = "modelo_entrenado"

# Cargar LabelEncoders, StandardScaler y umbral
label_encoders = joblib.load(os.path.join(output_dir, "label_encoders.pkl"))
scaler = joblib.load(os.path.join(output_dir, "scaler.pkl"))
threshold = np.load(os.path.join(output_dir, "threshold.npy"))

# Definir el modelo Autoencoder (debe coincidir con el modelo entrenado)
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=10):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(True),
            nn.Dropout(0.2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Cargar el modelo entrenado
model_path = os.path.join(output_dir, "autoencoder_model.pth")
input_dim = scaler.mean_.shape[0]  # Obtener la dimensión de entrada desde el scaler
autoencoder = Autoencoder(input_dim)
autoencoder.load_state_dict(torch.load(model_path))
autoencoder.eval()


# Aplicar Label Encoding a las columnas categóricas (Manejo de valores desconocidos con -1)
for column in datos_encuesta.columns:
    if column in label_encoders:  # Solo si la columna tiene un LabelEncoder guardado
        le = label_encoders[column]
        datos_encuesta[column] = datos_encuesta[column].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

# Aplicar escalado
datos_encuesta_scaled = scaler.transform(datos_encuesta)

# Convertir a tensor
datos_encuesta_tensor = torch.tensor(datos_encuesta_scaled, dtype=torch.float32)

# Realizar la reconstrucción con el autoencoder
with torch.no_grad():
    reconstruida = autoencoder(datos_encuesta_tensor)

# Calcular error de reconstrucción
error_reconstruccion = torch.mean((datos_encuesta_tensor - reconstruida) ** 2).item()

# Determinar si es un outlier
if error_reconstruccion > threshold:
    datos_encuesta["Legitimidad"] = "No legítimo"
else:
    datos_encuesta["Legitimidad"] = "Legítimo"

if st.button("Guardar Encuesta"):
    guardar_respuesta(datos_encuesta)
