import streamlit as st
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