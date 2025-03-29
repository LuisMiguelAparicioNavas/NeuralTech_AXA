import streamlit as st
import time
import pandas as pd
import os

# Ruta del archivo de usuarios dentro de la carpeta 'app/Data'
USERS_FILE = './Data/usuarios.csv'

# Inicializar variables de sesión
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = None
if 'user_type' not in st.session_state:
    st.session_state['user_type'] = None

# Splash Screen
if 'splash_shown' not in st.session_state:
    # Ruta de la imagen dentro de 'app/Data'
    st.image('./Data/AxaLogo.png', use_container_width=True)
    time.sleep(2)
    st.session_state.splash_shown = True
    st.rerun()  # Reemplazado

# Página de bienvenida después del Splash
st.title("Inicio de Sesión")
# Cambiar 'usuario' a 'correo' para pedir el correo electrónico
email = st.text_input("Correo electrónico")
password = st.text_input("Contraseña", type="password")

col1, col2 = st.columns(2)
if col1.button("Iniciar sesión"):
    users_df = pd.read_csv(USERS_FILE)  # Recargar datos por si hubo cambios
    # Buscar el correo electrónico en lugar de 'usuario'
    user_row = users_df[(users_df["Correo"] == email) & (users_df["Contraseña"] == password)]  # Cambio aquí

    # Verificar si user_row no está vacío antes de acceder
    if not user_row.empty:
        st.session_state['logged_in'] = True
        st.session_state['user_id'] = user_row.iloc[0]['ID']
        st.session_state['user_type'] = user_row.iloc[0]['Tipo']  # Usar 'Tipo' en mayúsculas
    else:
        st.error("Correo electrónico o contraseña incorrectos")

    if st.session_state.logged_in == True and st.session_state.user_type == 1:  # Trabajador
        st.switch_page("pages/empleado.py")
    elif st.session_state.logged_in == True and st.session_state.user_type == 0:  # Usuario
        st.switch_page("pages/encuesta.py")