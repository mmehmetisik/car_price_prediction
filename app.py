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

# Model ve bileşenleri yükleme
try:
    with open('models/final_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('models/feature_columns.pkl', 'rb') as file:
        feature_columns = pickle.load(file)
    with open('models/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
        
    # Ana başlık
    st.title('🚗 Araç Fiyat Tahmini')
    st.markdown("---")
    
    # Marka, renk ve eyalet seçeneklerini feature_columns'dan al
    brand_options = [col.replace('brand_', '') for col in feature_columns if col.startswith('brand_')]
    color_options = [col.replace('color_', '') for col in feature_columns if col.startswith('color_')]
    state_options = [col.replace('state_', '') for col in feature_columns if col.startswith('state_')]
    
    # Kullanıcı girdileri
    col1, col2 = st.columns(2)
    
    with col1:
        brand = st.selectbox('Marka', brand_options)
        year = st.slider('Model Yılı', 2000, 2024, 2020)
        mileage = st.number_input('Kilometre', min_value=0, max_value=300000, value=50000, step=1000)
    
    with col2:
        color = st.selectbox('Renk', color_options)
        title_status = st.selectbox('Araç Durumu', ['clean vehicle'])
        state = st.selectbox('Eyalet', state_options)
    
    if st.button('Fiyat Tahmini Yap', type='primary'):
        # Feature sözlüğü oluştur
        data_dict = {col: 0 for col in feature_columns}
        
        # Temel hesaplamalar
        car_age = 2024 - year
        avg_km_per_year = mileage / car_age if car_age > 0 else 0
        
        # Nümerik değerleri ölçekle
        numeric_features = ['year', 'mileage', 'car_age', 'avg_km_per_year', 'price_per_km']
        numeric_data = pd.DataFrame([[year, mileage, car_age, avg_km_per_year, 0]], 
                                  columns=numeric_features)
        scaled_numeric = scaler.transform(numeric_data)
        
        # Ölçeklenmiş değerleri sözlüğe ekle
        for i, feature in enumerate(numeric_features):
            data_dict[feature] = scaled_numeric[0][i]
        
        # Kategorik değişkenleri ayarla
        data_dict[f'brand_{brand}'] = 1
        data_dict[f'color_{color}'] = 1
        data_dict[f'state_{state}'] = 1
        data_dict['title_status_clean vehicle'] = 1
        data_dict['is_premium_1'] = 1 if brand in ['bmw', 'mercedes-benz'] else 0
        data_dict['is_popular_color_1'] = 1 if color in ['white', 'black', 'silver', 'gray'] else 0
        data_dict['clean_title_score_1'] = 1
        
        # Feature sırasını kontrol et ve düzelt
        input_df = pd.DataFrame([data_dict])[feature_columns]
        
        # Debug bilgisi
        st.write("Debug Bilgileri:")
        st.write("Input DataFrame kolonları:", input_df.columns.tolist())
        st.write("Beklenen kolonlar:", feature_columns)
        
        # Tahmin
        try:
            prediction = model.predict(input_df)[0]
            st.success(f'Tahmini Fiyat: ${prediction:,.2f}')
            
            # Detaylı bilgiler
            st.markdown("---")
            st.markdown("### Fiyatı Etkileyen Faktörler")
            info_col1, info_col2 = st.columns(2)
            
            with info_col1:
                st.write(f"- Araç Yaşı: {car_age} yıl")
                st.write(f"- Kilometre: {mileage:,} km")
                st.write(f"- Yıllık Ort. Kilometre: {avg_km_per_year:,.0f} km")
            
            with info_col2:
                st.write(f"- Premium Marka: {'Evet' if data_dict['is_premium_1'] == 1 else 'Hayır'}")
                st.write(f"- Popüler Renk: {'Evet' if data_dict['is_popular_color_1'] == 1 else 'Hayır'}")
                st.write(f"- Araç Durumu: {title_status}")
                
        except Exception as e:
            st.error(f"Tahmin işlemi sırasında hata: {str(e)}")
            st.write("Input DataFrame:", input_df)
            
except Exception as e:
    st.error(f"Uygulama başlatılırken hata: {str(e)}")
