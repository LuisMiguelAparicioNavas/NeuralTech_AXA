import streamlit as st
import pandas as pd
from streamlit_extras.card import card

# Cargar datos
df = pd.read_csv("./pages/respuestas_encuesta.csv", encoding="utf-8")

st.title("Respuestas de Encuesta")

# Mostrar cada fila en tarjetas
for _, row in df.iterrows():
    color = "#ffcccc" if row["Legitimidad"] == "No legítimo" else "#ccffcc"
    
    with st.container():
        st.markdown(
            f"""
            <div style="background-color: {color}; padding: 10px; border-radius: 10px; margin-bottom: 10px;">
                <b>ID_usuario:</b> {row['ID']}<br>
                <b>Centro Médico:</b> {row['Centro_Medico']}<br>
                <b>Diagnóstico:</b> {row['Diagnóstico']}<br>
                <b>Edad:</b> {row['Edad']}<br>
                <b>Duración Enfermedad:</b> {row['Duración_Enfermedad']}<br>
                <b>Síntomas:</b> {row['Síntomas']}<br>
                <b>Género:</b> {row['Género']}<br>
                <b>Historial Médico:</b> {row['Historial_Médico']}<br>
                <b>Motivo Consulta:</b> {row['Motivo_Consulta']}<br>
                <b>Tratamiento Recomendado:</b> {row['Tratamiento_Recomendado']}<br>
                <b>Actividad Laboral:</b> {row['Actividad_Laboral']}<br>
                <b>Gravedad:</b> {row['Gravedad']}<br>
                <b>Sesiones Necesarias:</b> {row['Sesiones_Necesarias']}<br>
                <b>Sesiones Solicitadas:</b> {row['Sesiones_Solicitadas']}<br>
                <b>Discrepancia Sesiones:</b> {row['Discrepancia_Sesiones']}<br>
                <b>Legitimidad:</b> {row['Legitimidad']}<br>
            </div>
            """,
            unsafe_allow_html=True
        )
