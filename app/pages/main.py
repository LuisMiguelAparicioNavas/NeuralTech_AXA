import streamlit as st
import time
import pandas as pd

# Ruta del archivo de usuarios dentro de la carpeta 'app/Data'
USERS_FILE = '../Data/usuarios.csv'

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
    st.image('../Data/AxaLogo.png', use_container_width=True)
    time.sleep(2)
    st.session_state.splash_shown = True
    st.experimental_rerun()  # Esto recarga la aplicación

# Página de bienvenida después del Splash
if not st.session_state['logged_in']:
    # Mostrar la pantalla de inicio de sesión
    st.title("Inicio de Sesión")
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
            # Aquí no necesitamos st.experimental_rerun(), el contenido cambiará automáticamente
        else:
            st.error("Correo electrónico o contraseña incorrectos")

# Si el usuario ha iniciado sesión y es trabajador (user_type == 1)
if st.session_state.logged_in and st.session_state.user_type == 1:
    # Mostrar la interfaz de 'main_employee' directamente en este archivo
    user = pd.read_csv("../Data/usuarios.csv")
    user = user[user["ID"] == st.session_state.user_id]

    st.image("../static/imgs/image.png", width=100)
    st.title("Resultados de las encuestas")

    # Aquí puedes mostrar los resultados de las encuestas u otros elementos
    # Agregar los componentes de la página de trabajador aquí
    # ...
