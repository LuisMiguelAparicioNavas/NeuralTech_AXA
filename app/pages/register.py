import streamlit as st
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