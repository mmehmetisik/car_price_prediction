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

# Model ve bileşenleri yükleme fonksiyonu
@st.cache_resource
def load_model_and_components():
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

# Ana başlık
st.title('🚗 Araç Fiyat Tahmini')
st.markdown("---")

# Yan panel
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

try:
    # Model ve bileşenleri yükle
    model, feature_columns, scaler = load_model_and_components()
    
    # Marka, renk ve eyalet seçeneklerini feature_columns'dan al
    brand_features = sorted([col.replace('brand_', '') for col in feature_columns if col.startswith('brand_')])
    color_features = sorted([col.replace('color_', '') for col in feature_columns if col.startswith('color_')])
    state_features = sorted([col.replace('state_', '') for col in feature_columns if col.startswith('state_')])
    
    # Ana panel - Kullanıcı girdileri
    st.header('Araç Özelliklerini Giriniz')
    
    # 2 sütunlu layout
    col1, col2 = st.columns(2)
    
    # İlk sütun
    with col1:
        brand = st.selectbox('Marka', brand_features)
        year = st.slider('Model Yılı', 2000, 2024, 2020)
        mileage = st.number_input('Kilometre', min_value=0, max_value=300000, value=50000, step=1000)
    
    # İkinci sütun
    with col2:
        color = st.selectbox('Renk', color_features)
        title_status = st.selectbox('Araç Durumu', ['clean vehicle'])
        state = st.selectbox('Eyalet', state_features)
    
    # Tahmin butonu
    if st.button('Fiyat Tahmini Yap', type='primary'):
        # Boş DataFrame oluştur
        input_data = pd.DataFrame(columns=feature_columns, index=[0])
        input_data.fillna(0, inplace=True)
        
        # Temel özellikleri hesapla
        car_age = 2024 - year
        avg_km_per_year = mileage / car_age if car_age > 0 else 0
        
        # Nümerik değerleri ata
        numeric_input = pd.DataFrame({
            'year': [year],
            'mileage': [mileage],
            'car_age': [car_age],
            'avg_km_per_year': [avg_km_per_year],
            'price_per_km': [0]  # Başlangıç değeri
        })
        
        # Nümerik değerleri ölçekle
        numeric_cols = ['year', 'mileage', 'car_age', 'avg_km_per_year', 'price_per_km']
        scaled_numeric = scaler.transform(numeric_input)
        
        # Ölçeklenmiş değerleri input_data'ya aktar
        for idx, col in enumerate(numeric_cols):
            input_data[col] = scaled_numeric[0][idx]
        
        # Kategorik değerleri ata
        input_data[f'brand_{brand}'] = 1
        input_data[f'color_{color}'] = 1
        input_data[f'state_{state}'] = 1
        input_data['title_status_clean vehicle'] = 1
        
        # Özel flagler
        input_data['is_premium_1'] = 1 if brand in ['bmw', 'mercedes-benz'] else 0
        input_data['is_popular_color_1'] = 1 if color in ['white', 'black', 'silver', 'gray'] else 0
        input_data['clean_title_score_1'] = 1
        
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
    st.error("Lütfen tüm değerleri doğru şekilde giriniz.")
