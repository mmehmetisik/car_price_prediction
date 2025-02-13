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
model_path = os.path.join('models', 'final_model.pkl')
with open(model_path, 'rb') as file:
   model = pickle.load(file)

feature_columns_path = os.path.join('models', 'feature_columns.pkl')
with open(feature_columns_path, 'rb') as file:
   feature_columns = pickle.load(file)

scaler_path = os.path.join('models', 'scaler.pkl')
with open(scaler_path, 'rb') as file:
   scaler = pickle.load(file)

# Ana başlık
st.title('🚗 Araç Fiyat Tahmini')
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
   brand = st.selectbox('Marka', ['ford', 'chevrolet', 'toyota', 'honda', 'bmw', 'nissan', 'dodge', 'mercedes-benz'])
   year = st.slider('Model Yılı', 2000, 2024, 2020)
   mileage = st.number_input('Kilometre', min_value=0, max_value=300000, value=50000, step=1000)

# İkinci sütun
with col2:
   color = st.selectbox('Renk', ['white', 'black', 'silver', 'gray', 'blue', 'red'])
   title_status = st.selectbox('Araç Durumu', ['clean vehicle', 'salvage insurance loss'])
   state = st.selectbox('Eyalet', ['california', 'florida', 'texas', 'new york', 'pennsylvania'])

# Tahmin butonu
if st.button('Fiyat Tahmini Yap', type='primary'):
   try:
       # İlk DataFrame oluşturma
       input_data = pd.DataFrame(index=[0])
       
       # Nümerik özellikleri ekle
       input_data['year'] = year
       input_data['mileage'] = mileage
       input_data['car_age'] = 2024 - year
       input_data['avg_km_per_year'] = mileage / input_data['car_age']
       input_data['price_per_km'] = 0.5  # Varsayılan değer
       
       # Nümerik değerleri ölçeklendir
       numeric_cols = ['year', 'mileage', 'car_age', 'avg_km_per_year', 'price_per_km']
       input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
       
       # Diğer özellikleri ekle
       # Brand encoding
       brand_cols = [col for col in feature_columns if col.startswith('brand_')]
       for col in brand_cols:
           brand_name = col.split('_')[1]
           input_data[col] = 1 if brand == brand_name else 0
           
       # Title status encoding
       input_data['title_status_clean vehicle'] = 1 if title_status == 'clean vehicle' else 0
       
       # Color encoding
       color_cols = [col for col in feature_columns if col.startswith('color_')]
       for col in color_cols:
           color_name = col.split('_')[1]
           input_data[col] = 1 if color == color_name else 0
           
       # State encoding
       state_cols = [col for col in feature_columns if col.startswith('state_')]
       for col in state_cols:
           state_name = col.split('_')[1]
           input_data[col] = 1 if state == state_name else 0
           
       # Price segment ve mileage segment (varsayılan değerler)
       segment_cols = [col for col in feature_columns if col.startswith('price_segment_') or col.startswith('mileage_segment_')]
       for col in segment_cols:
           input_data[col] = 0
           
       # Premium marka ve popular color flag'leri
       input_data['is_premium_1'] = 1 if brand in ['bmw', 'mercedes-benz'] else 0
       input_data['is_popular_color_1'] = 1 if color in ['white', 'black', 'silver', 'gray'] else 0
       input_data['clean_title_score_1'] = 1 if title_status == 'clean vehicle' else 0
       
       # Feature sırasını düzenle
       final_input = input_data[feature_columns]
       
       # Tahmin
       prediction = model.predict(final_input)[0]
       
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
