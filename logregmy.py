import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

st.write("""
# Logictic regression
""")

st.sidebar.header('Пользовательская настройка')
st.sidebar.write("""
# Dataset должен быть оцифрован.
Target должен быть последним столбцом.
""")

st.sidebar.header('Dataset нормирован?')
norm = st.sidebar.selectbox('Выбери',('Да','Нет'))

uploaded_file = st.sidebar.file_uploader('Загрузить свой Dataset', type=['csv'])
uploaded_df = pd.read_csv(uploaded_file)

st.sidebar.header('Количество итераций')
n_inputs = st.sidebar.slider('Количество итераций', 20,100000,1000)

st.sidebar.header('Выбери learning rate')
learning_rate = st.sidebar.slider('learning rate', 0.0001,1.0,0.1)

if norm == 'Нет':
    ss = StandardScaler()
    X = ss.fit_transform(uploaded_df.iloc[:, [0,-2]])
    y = uploaded_df.iloc[:, -1]
else:
    X = uploaded_df.iloc[:, [0,-2]]
    y = uploaded_df.iloc[:, -1]


def sigmoid(x):
        return 1 / (1 + np.exp(-x)) 


def fit( X, y):

    coef = np.random.normal(size=X.shape[1])     #  (это столбцы - 1)   кол-во фичей    (2;1)
    intercept_ = np.random.normal()              # w_0  свободные члены    #просто ичсло
    X = np.array(X)                                   # или len(X) ??# кол-во квартир - те строк в нашем датасете  # кол-во вводов - это длина (x1,x2,x3...)
    y = np.array(y).ravel()
    n_epochs = 10000

        # Градиент спуска для логистической регрессии
                   
    for i in range(n_epochs):
        z = np.dot(X, coef) + intercept_          
        y_pred = sigmoid(z)    

        gradient_coef=  X *(y_pred - y).reshape(-1, 1)
        gradient_intercept_ = (y_pred - y).mean()            
        coef = coef - learning_rate * gradient_coef.mean(axis=0)
        intercept_ = intercept_ - learning_rate * gradient_intercept_

    return y_pred


st.subheader('Веса для ваших данных')
st.write({uploaded_df.columns.tolist()[i] : fit(X, y)[i] for i in range(X.shape[1])})
st.write({'intercept' : fit(X, y)[-1]})

st.write("""
### Выбор параметров feature 
""")
st.subheader('first feature')
F1 = st.sidebar.selectbox(uploaded_df.columns.tolist()[0], uploaded_df.columns.tolist()[:-1])

st.subheader('second feature')
F2 = st.sidebar.selectbox(uploaded_df.columns.tolist()[1], uploaded_df.columns.tolist()[:-1])

st.subheader('Your plot')
plt.scatter(uploaded_df[uploaded_df.iloc[:, -1] == 0][F1], uploaded_df[uploaded_df.iloc[:, -1] == 0][F2], color='blue', label='0')
plt.scatter(uploaded_df[uploaded_df.iloc[:, -1] == 1][F1], uploaded_df[uploaded_df.iloc[:, -1] == 1][F2], color='red', label='1')
plt.xlabel(F1)
plt.ylabel(F2)
plt.legend()
plt.show()
st.pyplot(plt.gcf())
