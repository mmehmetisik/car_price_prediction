import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Model yükleme
model_path = os.path.join('models', 'final_model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Streamlit uygulama başlığı
st.title('Araç Fiyat Tahmin Uygulaması')

# Kullanıcı girdileri için form
st.header('Araç Özelliklerini Giriniz')

col1, col2, col3 = st.columns(3)

with col1:
    brand = st.selectbox('Marka', ['Ford', 'Toyota', 'BMW', 'Honda'])  # Markaları veri setinden alacağız
    year = st.number_input('Model Yılı', min_value=1990, max_value=2024, value=2020)
    mileage = st.number_input('Kilometre', min_value=0, value=50000)

with col2:
    color = st.selectbox('Renk', ['Siyah', 'Beyaz', 'Gri', 'Mavi'])  # Renkleri veri setinden alacağız
    state = st.selectbox('Eyalet', ['California', 'Texas', 'Florida'])  # Eyaletleri veri setinden alacağız

with col3:
    title_status = st.selectbox('Araç Durumu', ['Clean', 'Salvage'])
