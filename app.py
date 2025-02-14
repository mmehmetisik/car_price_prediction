import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Uygulama Konfigürasyonu
st.set_page_config(
    page_title="Araç Fiyat Tahmini",
    page_icon="🚗",
    layout="wide"
)

# Model, scaler ve feature columns yükleme
@st.cache_resource
def load_model_components():
    model_path = os.path.join('models', 'final_model.pkl')
    feature_columns_path = os.path.join('models', 'feature_columns.pkl')
    scaler_path = os.path.join('models', 'scaler.pkl')
    
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    with open(feature_columns_path, 'rb') as file:
        feature_columns = pickle.load(file)
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    return model, feature_columns, scaler

model, feature_columns, scaler = load_model_components()

# Ana başlık
st.title('🚗 Araç Fiyat Tahmin Uygulaması')
st.markdown("---")

# Yan panel (sidebar) için açıklama
st.sidebar.header("Uygulama Hakkında")
st.sidebar.markdown("""
### Araç Fiyat Tahmin Sistemi
Bu uygulama, gelişmiş makine öğrenmesi algoritmaları kullanarak araç fiyat tahmini yapar.

**Özellikler:**
- Gerçek piyasa verilerine dayalı tahminler
- 50'den fazla özellik analizi
- Anlık fiyat hesaplama
- Detaylı araç özellikleri analizi

**Veri Kaynağı:** 
- USA Cars Dataset
- 2,500+ araç verisi
- Güncel piyasa analizi
""")

# Ana panel - Kullanıcı girdileri
st.header('Araç Özelliklerini Giriniz')

# 2 sütunlu layout
col1, col2 = st.columns(2)

# İlk sütun
with col1:
    brand = st.selectbox('Marka', ['bmw', 'chevrolet', 'chrysler', 'dodge', 'ford', 'gmc', 'hyundai', 'jeep', 'nissan'])
    year = st.slider('Model Yılı', 2000, 2024, 2020)
    mileage = st.number_input('Kilometre', min_value=0, max_value=300000, value=50000, step=1000)

# İkinci sütun
with col2:
    color = st.selectbox('Renk', ['black', 'blue', 'gray', 'no_color', 'red', 'silver', 'white'])
    title_status = st.selectbox('Araç Durumu', ['clean vehicle'])
    state = st.selectbox('Eyalet', ['arizona', 'california', 'florida', 'texas', 'new york', 'pennsylvania'])

# Tahmin butonu
if st.button('Fiyat Tahmini Yap', type='primary'):
    try:
        # Nümerik özellikleri hazırla
        car_age = 2024 - year
        avg_km_per_year = mileage / car_age if car_age > 0 else 0
        price_per_km = 0  # Başlangıç değeri
        
        # Nümerik değerleri ölçekle
        numeric_features = pd.DataFrame({
            'year': [year],
            'mileage': [mileage],
            'car_age': [car_age],
            'avg_km_per_year': [avg_km_per_year],
            'price_per_km': [price_per_km]
        })
        
        scaled_numeric = scaler.transform(numeric_features)
        
        # Tüm özellikleri içeren boş DataFrame oluştur
        input_data = pd.DataFrame(columns=feature_columns, index=[0])
        input_data = input_data.fillna(0)
        
        # Ölçeklenmiş nümerik değerleri ekle
        numeric_cols = ['year', 'mileage', 'car_age', 'avg_km_per_year', 'price_per_km']
        for i, col in enumerate(numeric_cols):
            input_data[col] = scaled_numeric[0][i]
        
        # Kategorik değerleri ekle
        input_data[f'brand_{brand}'] = 1
        input_data[f'color_{color}'] = 1
        input_data[f'state_{state}'] = 1
        input_data['title_status_clean vehicle'] = 1
        
        # Özel flagler
        input_data['is_premium_1'] = 1 if brand in ['bmw', 'mercedes-benz'] else 0
        input_data['is_popular_color_1'] = 1 if color in ['white', 'black', 'silver', 'gray'] else 0
        input_data['clean_title_score_1'] = 1 if title_status == 'clean vehicle' else 0
        
        # Tahmin
        prediction = model.predict(input_data)[0]
        
        # Sonuç gösterimi
        st.success(f'Tahmini Fiyat: ${prediction:,.2f}')
        
        # Detaylı açıklama
        st.markdown("---")
        st.markdown("### Fiyatı Etkileyen Faktörler")
        detail_col1, detail_col2 = st.columns(2)
        
        with detail_col1:
            st.write(f"- Araç Yaşı: {car_age} yıl")
            st.write(f"- Kilometre: {mileage:,} km")
            st.write(f"- Yıllık Ort. Kilometre: {avg_km_per_year:,.0f} km")
            
        with detail_col2:
            st.write(f"- Premium Marka: {'Evet' if input_data['is_premium_1'].iloc[0] == 1 else 'Hayır'}")
            st.write(f"- Popüler Renk: {'Evet' if input_data['is_popular_color_1'].iloc[0] == 1 else 'Hayır'}")
            st.write(f"- Araç Durumu: {title_status}")
            
    except Exception as e:
        st.error(f"Bir hata oluştu: {str(e)}")
        st.error("Hata detayları için:")
        st.write("Feature sırası:", feature_columns)
        st.write("Input data kolonları:", input_data.columns.tolist())
