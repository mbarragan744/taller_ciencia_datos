import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# -----------------------------
# Cargar modelo y objetos
# -----------------------------

model = load_model("modelo_credito.h5")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
columns = joblib.load("columns.pkl")

st.title("Predicción del Puntaje de Crédito")

st.write("Ingrese la información financiera del cliente")

# -----------------------------
# Inputs del usuario (EN ESPAÑOL)
# -----------------------------

age = st.number_input("Edad del cliente", 18, 100)

annual_income = st.number_input("Ingreso anual del cliente")

monthly_salary = st.number_input("Salario mensual neto")

num_bank_accounts = st.number_input("Número de cuentas bancarias", 0, 20)

num_credit_card = st.number_input("Número de tarjetas de crédito", 0, 20)

interest_rate = st.number_input("Tasa de interés de los préstamos")

num_loans = st.number_input("Número de préstamos")

delay_due_date = st.number_input("Días de retraso en pagos")

num_delayed_payment = st.number_input("Cantidad de pagos atrasados")

num_credit_inquiries = st.number_input("Número de consultas de crédito")

outstanding_debt = st.number_input("Deuda pendiente")

credit_utilization = st.number_input("Porcentaje de uso del crédito")

credit_history_age = st.number_input("Antigüedad del historial crediticio")

total_emi = st.number_input("Pago mensual total de préstamos")

amount_invested = st.number_input("Cantidad invertida mensualmente")

monthly_balance = st.number_input("Saldo mensual en cuentas")

# -----------------------------
# Botón predicción
# -----------------------------

if st.button("Predecir puntaje de crédito"):

    input_dict = {
        "Age": age,
        "Annual_Income": annual_income,
        "Monthly_Inhand_Salary": monthly_salary,
        "Num_Bank_Accounts": num_bank_accounts,
        "Num_Credit_Card": num_credit_card,
        "Interest_Rate": interest_rate,
        "Num_of_Loan": num_loans,
        "Delay_from_due_date": delay_due_date,
        "Num_of_Delayed_Payment": num_delayed_payment,
        "Num_Credit_Inquiries": num_credit_inquiries,
        "Outstanding_Debt": outstanding_debt,
        "Credit_Utilization_Ratio": credit_utilization,
        "Credit_History_Age": credit_history_age,
        "Total_EMI_per_month": total_emi,
        "Amount_invested_monthly": amount_invested,
        "Monthly_Balance": monthly_balance
    }

    # Crear dataframe
    input_df = pd.DataFrame([input_dict])

    # Ajustar columnas faltantes
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Escalar datos
    scaled_data = scaler.transform(input_df)

    # Aplicar PCA
    pca_data = pca.transform(scaled_data)

    # Predicción
    prediction = model.predict(pca_data)

    clase = np.argmax(prediction)

    st.subheader("Resultado del modelo")

    if clase == 0:
        st.success("Riesgo Bajo")

    elif clase == 1:
        st.warning("Riesgo Medio")

    else:
        st.error("Riesgo Alto")

    st.write("Probabilidades de cada categoría:")
    st.write(prediction)
    
    prob_bajo = prediction[0][0]
prob_medio = prediction[0][1]
prob_alto = prediction[0][2]

st.write("Probabilidad Riesgo Bajo:", round(prob_bajo*100,2), "%")
st.write("Probabilidad Riesgo Medio:", round(prob_medio*100,2), "%")

st.write("Probabilidad Riesgo Alto:", round(prob_alto*100,2), "%")

