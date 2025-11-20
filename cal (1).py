import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Predicción de calificación  ''')
st.image("estudio.jpg", caption="Predicción de la calificación.")

st.header('Datos personales')

def user_input_features():
  # Entrada
  Horas_estudiadas = st.number_input('hours_studied:', min_value=0, max_value=100, value = 0, step = 1)
  Horas_dormidas = st.number_input('sleep_hours:',  min_value=0, max_value=1, value = 0, step = 1)
  Asistencia = st.number_input('attendance_percent:', min_value=0, max_value=230, value = 0, step = 1)
  Previo = st.number_input('previous_scores:', min_value=0, max_value=140, value = 0, step = 1)


  user_input_data = {'hours_studied': Horas_estudiadas,
                     'sleep_hours': Horas_dormidas,
                     'attendance_percent': Asistencia,
                     'previous_scores': Previo,
                     }

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

datos =  pd.read_csv('Examen.csv', encoding='latin-1')
X = datos.drop(columns='exam_score')
y = datos['exam_score']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1614372)
LR = LinearRegression()
LR.fit(X_train,y_train)

b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*df['hours_studied'] + b1[1]*df['sleep_hours'] + b1[2]*df['attendance_percent'] + b1[3]*df['previous_scores']

st.subheader('Cálculo de la calificación')
st.write('La calificación es ', prediccion)
