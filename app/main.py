import streamlit as st
import time
import pandas as pd
import os

# Ruta del archivo de usuarios dentro de la carpeta 'app/Data'
USERS_FILE = './Data/usuarios.csv'

# Splash Screen
if 'splash_shown' not in st.session_state:
    # Ruta de la imagen dentro de 'app/Data'
    st.image('./Data/AxaLogo.png', use_column_width=True)
    time.sleep(2)
    st.session_state.splash_shown = True
    st.rerun()  # Reemplazado

# Login Page
if 'logged_in' not in st.session_state:
    st.title("Inicio de Sesión")
    # Cambiar 'usuario' a 'correo' para pedir el correo electrónico
    email = st.text_input("Correo electrónico")
    password = st.text_input("Contraseña", type="password")
    
    col1, col2 = st.columns(2)
    if col1.button("Iniciar sesión"):
        users_df = pd.read_csv(USERS_FILE)  # Recargar datos por si hubo cambios
        # Buscar el correo electrónico en lugar de 'usuario'
        user_row = users_df[(users_df["Correo"] == email) & (users_df["Contraseña"] == password)]  # Cambio aquí
        print(user_row.columns)  # Ver las columnas disponibles

        # Verificar si user_row no está vacío antes de acceder
        if not user_row.empty:
            if 'ID' in user_row.columns:  # Verificar que la columna 'ID' existe
                st.session_state.logged_in = True
                st.session_state.user_id = user_row.iloc[0]['ID']  # Usar 'ID' en mayúsculas
                st.session_state.user_type = user_row.iloc[0]['Tipo']  # Usar 'Tipo' en mayúsculas
                st.rerun()  # Reemplazado
            else:
                st.error("La columna 'ID' no está presente en el archivo de usuarios.")
        else:
            st.error("Correo electrónico o contraseña incorrectos")
    
    if col2.button("Registrarse"):
        st.session_state.register = True
        st.rerun()  # Reemplazado

# Registration Page
if 'register' in st.session_state:
    st.title("Registro de Usuario")
    new_email = st.text_input("Correo electrónico")
    new_password = st.text_input("Nueva Contraseña", type="password")
    if st.button("Registrar"):
        users_df = pd.read_csv(USERS_FILE)  # Recargar datos por si hubo cambios
        # Verificar si el correo electrónico ya existe
        if new_email in users_df["Correo"].values:
            st.error("El correo electrónico ya está registrado")
        else:
            new_id = users_df["ID"].max() + 1 if not users_df.empty else 1  # Usar 'ID' en mayúsculas
            new_user = pd.DataFrame([[new_id, new_email, new_password, 0]], columns=["ID", "usuario", "Contraseña", "Tipo"])  # Usar 'ID' y 'Tipo' en mayúsculas
            users_df = pd.concat([users_df, new_user], ignore_index=True)
            users_df.to_csv(USERS_FILE, index=False)
            st.success("Registro exitoso")
            del st.session_state.register
            st.rerun()  # Reemplazado

# Cliente Page
if 'logged_in' in st.session_state and st.session_state.user_type == 0:
    st.title(f"Bienvenido Cliente (ID: {st.session_state.user_id})")

# Paciente Page
if 'logged_in' in st.session_state and st.session_state.user_type == 1:
    st.title(f"Bienvenido Paciente (ID: {st.session_state.user_id})")
