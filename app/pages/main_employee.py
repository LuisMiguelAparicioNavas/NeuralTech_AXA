import streamlit as st
import pandas as pd

user = pd.read_csv("../../Data/usuarios.csv")
user = user[user["ID"] == 1] #st.session_state.id

st.image("../static/imgs/image.png", width=100)

st.title("Resultados de las encuestas")

