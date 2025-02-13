import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Uygulama Stili
st.set_page_config(page_title="🚗 Araç Fiyat Tahmini", page_icon="🚘", layout="centered")

# Model ve Feature Listesini Yükleme
model_path = os.path.join(os.path.dirname(__file__), "models", "final_model.pkl")
feature_columns_path = os.path.join(os.path.dirname(__file__), "models", "feature_columns.pkl")

with open(model_path, "rb") as file:
    model = pickle.load(file)

with open(feature_columns_path, "rb") as file:
    feature_columns = pickle.load(file)

# Kullanıcı Arayüzü
st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🚗 Araç Fiyat Tahmini</h1>", unsafe_allow_html=True)
st.write("Lütfen aracın özelliklerini girerek tahmini fiyatı öğrenin.")

# Kullanıcı Girdileri
brand = st.selectbox("Marka", ["bmw", "chevrolet", "chrysler", "dodge", "ford", "gmc", "hyundai", "jeep", "nissan"])
year = st.number_input("Model Yılı", min_value=1980, max_value=2024, value=2015, step=1)
mileage = st.number_input("Kilometre", min_value=0, max_value=500000, value=100000, step=1000)
color = st.selectbox("Renk", ["black", "blue", "gray", "no_color", "red", "silver", "white"])
condition = st.selectbox("Araç Durumu", ["clean vehicle", "salvage", "rebuilt", "parts only", "damage"])

# **Tahmin Butonu**
if st.button("🚀 Tahmin Yap"):
    # Feature Listesine Göre DataFrame Oluştur
    input_data = pd.DataFrame(columns=feature_columns)
    
    # **Sayısal Değerler**
    input_data.loc[0, "year"] = year
    input_data.loc[0, "mileage"] = mileage
    input_data.loc[0, "car_age"] = 2024 - year  # Araç Yaşı
    input_data.loc[0, "avg_km_per_year"] = mileage / (2024 - year + 1)  # Yıllık Ortalama Km
    input_data.loc[0, "price_per_km"] = np.random.uniform(0.01, 0.05)  # Model için tahmini bir değer

    # **Kategorik Değişkenleri One-Hot Encoding’e Çevirme**
    input_data.loc[0, f"brand_{brand}"] = 1  # Marka
    input_data.loc[0, f"title_status_{condition}"] = 1  # Araç Durumu
    input_data.loc[0, f"color_{color}"] = 1  # Renk
    
    # **Eksik Feature'ları 0 ile Doldurma**
    input_data = input_data.fillna(0)

    # **Tahmin**
    prediction = model.predict(input_data)[0]
    
    st.success(f"💰 Tahmini Araç Fiyatı: **${prediction:,.2f}**")
