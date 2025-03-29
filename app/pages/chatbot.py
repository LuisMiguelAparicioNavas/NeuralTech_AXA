from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

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
