import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

path = 'modelo_falhas.pkl'
model = joblib.load(path)

# Função para fazer previsões
def fazer_previsao(features):
    # Adapte conforme necessário para o seu modelo
    previsao = model.predict(features.reshape(1, -1))
    return previsao[0]

feature_input = st.text_input('Insira os valores das características (separados por vírgula):')

if st.button('Prever'):
    # Converter entrada em uma lista de float
    features = [float(valor) for valor in feature_input.split(',')]

    # Fazer previsão
    resultado = fazer_previsao(np.array(features))

    # Exibir resultado
    if resultado == 1:
        st.write('Resultado da Detecção de Falhas: Falha detectada!')
    else:
        st.write('Resultado da Detecção de Falhas: Sem falha detectada.')
