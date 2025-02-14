import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Sayfa yapılandırması
st.set_page_config(
    page_title="Araç Fiyat Tahmini",
    page_icon="🚗",
    layout="wide"
)

# Başlık ve açıklama
st.title("🚗 Araç Fiyat Tahmin Uygulaması")
st.markdown("""
Bu uygulama, girdiğiniz araç özelliklerine göre tahmini bir fiyat sunmaktadır.
""")

# Model ve gerekli dosyaları yükleme
@st.cache_resource
def load_model_and_components():
    with open('models/final_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('models/feature_columns.pkl', 'rb') as file:
        features = pickle.load(file)
    with open('models/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, features, scaler

try:
    model, features, scaler = load_model_and_components()
    st.success('Model ve bileşenler başarıyla yüklendi! 🎉')
except Exception as e:
    st.error(f'Model yüklenirken bir hata oluştu: {str(e)}')
    st.stop()

# Marka seçenekleri (feature columns'dan çıkarılıyor)
brand_features = [col.replace('brand_', '') for col in features if col.startswith('brand_')]

# Renk seçenekleri
color_features = [col.replace('color_', '') for col in features if col.startswith('color_')]

# Eyalet seçenekleri
state_features = [col.replace('state_', '') for col in features if col.startswith('state_')]

# Kullanıcı girdileri için kolonlar
col1, col2 = st.columns(2)

with col1:
    st.subheader("Temel Bilgiler")
    brand = st.selectbox('Marka:', brand_features)
    year = st.number_input('Model Yılı:', min_value=1990, max_value=2024, value=2020)
    mileage = st.number_input('Kilometre:', min_value=0, max_value=500000, value=50000)
    color = st.selectbox('Renk:', color_features)

with col2:
    st.subheader("Ek Bilgiler")
    state = st.selectbox('Eyalet:', state_features)
    title_status = st.selectbox('Araç Durumu:', ['clean vehicle'])
    is_premium = 1 if brand.lower() in ['bmw', 'mercedes-benz', 'lexus', 'infiniti', 'maserati'] else 0
    is_popular_color = 1 if color.lower() in ['white', 'black', 'silver', 'gray'] else 0

# Tahmin butonu
if st.button('Fiyat Tahmini Yap'):
    try:
        # Feature hazırlama
        car_age = 2024 - year
        avg_km_per_year = mileage / car_age if car_age > 0 else 0
        price_per_km = 0  # Bu değer tahmin sonrası güncellenecek
        
        # Temel özellikleri DataFrame'e dönüştürme
        data = {
            'year': year,
            'mileage': mileage,
            'car_age': car_age,
            'avg_km_per_year': avg_km_per_year,
            'price_per_km': price_per_km
        }
        
        # One-hot encoding için tüm kolonları 0 ile doldurma
        for feature in features:
            if feature not in data:
                data[feature] = 0
                
        # Seçilen özellikleri 1 yapma
        data[f'brand_{brand.lower()}'] = 1
        data[f'color_{color.lower()}'] = 1
        data[f'state_{state.lower()}'] = 1
        data['title_status_clean vehicle'] = 1
        data['is_premium'] = is_premium
        data['is_popular_color'] = is_popular_color
        data['clean_title_score'] = 1 if title_status == 'clean vehicle' else 0
        
        # DataFrame oluşturma ve feature sırasını düzenleme
        input_df = pd.DataFrame([data])
        input_df = input_df[features]
        
        # Sayısal değişkenleri ölçekleme
        numeric_features = ['year', 'car_age', 'mileage', 'avg_km_per_year', 'price_per_km']
        input_df[numeric_features] = scaler.transform(input_df[numeric_features])
        
        # Tahmin yapma
        prediction = model.predict(input_df)[0]
        
        # Sonucu gösterme
        st.success(f'Tahmini Araç Fiyatı: ${prediction:,.2f}')
        
        # Detaylı bilgiler
        with st.expander("Detaylı Bilgiler"):
            st.write(f"Araç Yaşı: {car_age} yıl")
            st.write(f"Yıllık Ortalama KM: {avg_km_per_year:,.2f} km")
            st.write(f"Premium Marka: {'Evet' if is_premium else 'Hayır'}")
            st.write(f"Popüler Renk: {'Evet' if is_popular_color else 'Hayır'}")
            
    except Exception as e:
        st.error(f'Tahmin yapılırken bir hata oluştu: {str(e)}')

# Footer
st.markdown("""
---
📊 Bu uygulama, makine öğrenmesi kullanılarak geliştirilmiş bir fiyat tahmin modelidir.
""")
