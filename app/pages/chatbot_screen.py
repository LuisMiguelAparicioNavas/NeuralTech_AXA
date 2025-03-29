import streamlit as st
from chatbot import chatbot

# Configurar la página
st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="wide")

st.title("Chatbot")

datos_paciente = {
    "ID_Paciente": "PT-00001",
    "Centro_Medico": "Centro A",
    "Diagnóstico": "Artritis reumatoide",
    "Edad": 45,
    "Duración_Enfermedad": 34,
    "Síntomas": "Dolor en articulaciones",
    "Género": "No binario",
    "Historial_Médico": "Sin antecedentes",
    "Motivo_Consulta": "Empeoramiento reciente",
    "Tratamiento_Recomendado": "Terapia ocupacional",
    "Actividad_Laboral": "Jubilado",
    "Gravedad": "Moderada",
    "Sesiones_Necesarias": 8,
    "Fraude_Paciente": 0,
    "Fraude_Centro": 0,
    "Sesiones_Solicitadas": 9,
    "Discrepancia_Sesiones": 1,
    "Fraude_Total": 0
}


# Inicializar el estado de la sesión para almacenar el historial de mensajes
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar el historial de mensajes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada de usuario
if prompt := st.chat_input("Escribe un mensaje..."):
    # Agregar el mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Mostrar el mensaje del usuario en la interfaz
    with st.chat_message("user"):
        st.markdown(prompt)

    # Mostrar un indicador de carga
    with st.chat_message("assistant"):
        st.markdown("Cargando... ⏳")

    # Respuesta genérica del bot
    response = chatbot(datos_paciente, prompt)

    # Remover el mensaje de carga y agregar la respuesta del bot al historial
    st.session_state.messages[-1]["content"] = response

    # Mostrar la respuesta del bot en la interfaz
    with st.chat_message("assistant"):
        st.markdown(response)
