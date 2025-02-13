import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 📌 **Kaydedilen Modeli Yükle**
model_path = "models/final_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# 📌 **Eğitimde Kullanılan Sütunları Yükle**
feature_columns_path = "models/feature_columns.pkl"
with open(feature_columns_path, "rb") as file:
    feature_columns = pickle.load(file)

# 📌 **Streamlit Başlığı**
st.title("🚗 Araç Fiyat Tahmini Uygulaması 💰")
st.write("Lütfen aracın özelliklerini girerek tahmini fiyatı öğrenin.")

# 📌 **Kullanıcının Gireceği Değişkenler**
brand = st.selectbox("Marka", ["chevrolet", "dodge", "ford", "toyota"])
model_name = st.text_input("Model Adı (Opsiyonel)")
year = st.number_input("Model Yılı", min_value=1990, max_value=2025, step=1)
mileage = st.number_input("Kilometre", min_value=0, max_value=500000, step=1000)
color = st.selectbox("Renk", ["black", "silver", "blue", "red", "white"])
title_status = st.selectbox("Araç Durumu", ["clean vehicle", "salvage insurance loss"])

# 📌 **Kullanıcının Verisini DataFrame Formatına Getirme**
input_data = pd.DataFrame({
    "year": [year],
    "mileage": [mileage],
    "brand": [brand],
    "color": [color],
    "title_status": [title_status]
})

# 📌 **One-Hot Encoding Uygula**
input_data = pd.get_dummies(input_data)

# 📌 **Eksik Olan Sütunları Modelin Beklediği Formata Getirme**
for col in feature_columns:
    if col not in input_data.columns:
        input_data[col] = 0  # Eksik sütunları 0 ile doldur

# 📌 **Sütunları Modele Uygun Hale Getirme**
input_data = input_data[feature_columns]

# 📌 **Tahmin Butonu**
if st.button("Tahmin Yap"):
    prediction = model.predict(input_data)[0]
    st.success(f"🚘 Tahmini Araç Fiyatı: **${prediction:,.2f}**")
