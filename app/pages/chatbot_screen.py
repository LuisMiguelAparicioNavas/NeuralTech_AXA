import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

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

# Funcion que en base a un input del usuario y sus datos, te genera una respuesta
def chatbot(patient_data, user_input):
    template = """
    Eres un asistente médico experto. Con base en los siguientes datos del paciente, 
    responda la pregunta con recomendaciones y observaciones personalizadas.

    Patient Data:
    - ID_Paciente: {ID_Paciente}
    - Centro_Medico: {Centro_Medico}
    - Diagnóstico: {Diagnóstico}
    - Edad: {Edad}
    - Duración_Enfermedad: {Duración_Enfermedad}
    - Síntomas: {Síntomas}
    - Género: {Género}
    - Historial_Médico: {Historial_Médico}
    - Motivo_Consulta: {Motivo_Consulta}
    - Tratamiento_Recomendado: {Tratamiento_Recomendado}
    - Actividad_Laboral: {Actividad_Laboral}
    - Gravedad: {Gravedad}
    - Sesiones_Necesarias: {Sesiones_Necesarias}
    - Fraude_Paciente: {Fraude_Paciente}
    - Fraude_Centro: {Fraude_Centro}
    - Sesiones_Solicitadas: {Sesiones_Solicitadas}
    - Discrepancia_Sesiones: {Discrepancia_Sesiones}
    - Fraude_Total: {Fraude_Total}

    Question: {question}

    Answer:
    """

    model = OllamaLLM(model="llama3")
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    print("Welcome to the AI ChatBot! Type 'exit' to quit.")
    params = {**patient_data, "question": user_input}
    result = chain.invoke(params)
    return result


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


