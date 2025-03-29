import streamlit as st
import pandas as pd

user = pd.read_csv("../Data/usuarios.csv")
user = user[user["ID"] == st.session_state.user_id] 

st.image("../static/imgs/image.png", width=100)

st.title("Resultados de las encuestas")

