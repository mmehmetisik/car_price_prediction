import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Uygulama Stili
st.set_page_config(
    page_title="🚗 Araç Fiyat Tahmini",
    page_icon="🚘",
    layout="centered"
)

# Özel CSS ile UI'yi güzelleştirme
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
        }
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border-radius: 10px;
        }
        .stButton>button:hover {
            background-color: #ff1c1c;
        }
        .stTextInput>div>div>input {
            font-size: 16px;
        }
        .stSelectbox>div>div>select {
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Sayfa Başlığı
st.markdown(
    "<h1 style='text-align: center; color: #ff4b4b;'>🚗 Araç Fiyat Tahmini Uygulaması</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h4 style='text-align: center; color: #333;'>Lütfen aracın özelliklerini girerek tahmini fiyatı öğrenin.</h4>",
    unsafe_allow_html=True
)

st.write("")

# Modeli Yükleme
model_path = os.path.join(os.path.dirname(__file__), "models", "final_model.pkl")
feature_columns_path = os.path.join(os.path.dirname(__file__), "models", "feature_columns.pkl")

with open(model_path, "rb") as file:
    model = pickle.load(file)

with open(feature_columns_path, "rb") as file:
    feature_columns = pickle.load(file)

# Kullanıcı Girdileri
st.markdown("### 📌 Araç Bilgilerini Giriniz")

brand = st.selectbox("Marka", ["Toyota", "Ford", "Chevrolet", "Honda", "BMW", "Mercedes"])
model_name = st.text_input("Model Adı (Opsiyonel)")
year = st.number_input("Model Yılı", min_value=1980, max_value=2024, value=2015, step=1)
mileage = st.number_input("Kilometre", min_value=0, max_value=500000, value=100000, step=1000)
color = st.selectbox("Renk", ["Beyaz", "Siyah", "Gri", "Mavi", "Kırmızı", "Yeşil", "Sarı"])
condition = st.selectbox("Araç Durumu", ["Clean Vehicle", "Salvage", "Rebuilt", "Parts Only", "Damage"])

# Tahmin Butonu
if st.button("🚀 Tahmin Yap"):
    input_data = pd.DataFrame([[brand, model_name, year, mileage, color, condition]], columns=feature_columns)
    
    # Model ile Tahmin
    prediction = model.predict(input_data)[0]
    
    st.success(f"💰 Tahmini Araç Fiyatı: **${prediction:,.2f}**") 
