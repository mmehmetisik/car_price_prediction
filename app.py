import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import RobustScaler

# Uygulama Konfigürasyonu
st.set_page_config(
    page_title="Araç Fiyat Tahmini",
    page_icon="🚗",
    layout="wide"
)

# Model, scaler ve feature columns yükleme
@st.cache_resource
def load_model_and_features():
    with open("models/final_model.pkl", "rb") as file:
        model = pickle.load(file)
    with open("models/feature_columns.pkl", "rb") as file:
        feature_columns = pickle.load(file)
    # Scaler'ı yükle veya yeni bir tane oluştur
    scaler = RobustScaler()
    return model, feature_columns, scaler

model, feature_columns, scaler = load_model_and_features()

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

# 3 sütunlu layout
col1, col2, col3 = st.columns(3)

# İlk sütun
with col1:
    brand = st.selectbox('Marka', ['ford', 'chevrolet', 'toyota', 'honda', 'bmw', 'nissan', 'dodge', 'mercedes-benz'])
    model_name = st.text_input("Model Adı (Opsiyonel)")
    year = st.slider('Model Yılı', 2000, 2024, 2020)
    mileage = st.number_input('Kilometre', min_value=0, max_value=300000, value=50000, step=1000)

# İkinci sütun
with col2:
    color = st.selectbox('Renk', ['white', 'black', 'silver', 'gray', 'blue', 'red'])
    state = st.selectbox('Eyalet', ['california', 'florida', 'texas', 'new york', 'pennsylvania'])

# Üçüncü sütun
with col3:
    title_status = st.selectbox('Araç Durumu', ['clean vehicle', 'salvage insurance loss'])

# Tahmin butonu
if st.button('Fiyat Tahmini Yap', type='primary'):
    try:
        # Temel özellikleri DataFrame'e dönüştür
        input_data = pd.DataFrame({
            "year": [year],
            "mileage": [mileage],
            "brand": [brand],
            "color": [color],
            "title_status": [title_status],
            "state": [state]
        })

        # Türetilmiş özellikleri ekle
        input_data['car_age'] = 2024 - input_data['year']
        input_data['avg_km_per_year'] = input_data['mileage'] / input_data['car_age']
        input_data['is_premium'] = input_data['brand'].isin(['bmw', 'mercedes-benz']).astype(int)
        input_data['is_popular_color'] = input_data['color'].isin(['white', 'black', 'silver', 'gray']).astype(int)
        input_data['clean_title_score'] = (input_data['title_status'] == 'clean vehicle').astype(int)

        # Nümerik kolonları sakla
        numeric_cols = ['year', 'mileage', 'car_age', 'avg_km_per_year']
        numeric_data = input_data[numeric_cols].copy()

        # One-Hot Encoding uygula
        input_data = pd.get_dummies(input_data, columns=['brand', 'color', 'title_status', 'state'])

        # Eksik kolonları ekle
        for col in feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0

        # Nümerik kolonları ölçeklendir
        input_data[numeric_cols] = scaler.fit_transform(numeric_data)

        # Sütunları modele uygun hale getir
        input_data = input_data[feature_columns]

        # Tahmin yap
        prediction = model.predict(input_data)[0]

        # Sonuç gösterimi
        st.success(f'Tahmini Fiyat: ${prediction:,.2f}')

        # Detaylı açıklama
        st.markdown("---")
        st.markdown("### Fiyatı Etkileyen Faktörler")
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"- Araç Yaşı: {2024 - year} yıl")
            st.write(f"- Kilometre: {mileage:,} km")
            st.write(f"- Premium Marka: {'Evet' if brand in ['bmw', 'mercedes-benz'] else 'Hayır'}")

        with col2:
            st.write(f"- Durum: {title_status}")
            st.write(f"- Lokasyon: {state}")
            st.write(f"- Renk: {color}")

    except Exception as e:
        st.error(f"Bir hata oluştu: {str(e)}")
        st.error("Lütfen tüm alanları doğru şekilde doldurunuz.")
