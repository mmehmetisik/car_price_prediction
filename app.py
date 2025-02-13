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

# Model ve feature columns yükleme
model_path = os.path.join('models', 'final_model.pkl')
with open(model_path, 'rb') as file:
   model = pickle.load(file)

# Feature columns'ı yükle
with open('feature_columns.pkl', 'rb') as file:
   feature_columns = pickle.load(file)

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

def prepare_features(brand, year, mileage, color, state, title_status):
   # Boş DataFrame'i sadece model feature'ları ile oluştur
   features = pd.DataFrame(0, index=[0], columns=feature_columns)
   
   # Temel özellikleri ekle
   features['year'] = year
   features['mileage'] = mileage
   features['car_age'] = 2024 - year
   features['avg_km_per_year'] = mileage / (2024 - year)
   features['price_per_km'] = 0
   
   # Kategorik değişkenleri kontrol edip ekle
   brand_col = f'brand_{brand.lower()}'
   if brand_col in feature_columns:
       features[brand_col] = 1
       
   color_col = f'color_{color.lower()}'
   if color_col in feature_columns:
       features[color_col] = 1
       
   state_col = f'state_{state.lower()}'
   if state_col in feature_columns:
       features[state_col] = 1
       
   # Title status için özel kontrol
   title_col = f'title_status_{title_status}'
   if title_col in feature_columns:
       features[title_col] = 1
   
   # Premium marka kontrolü
   if 'is_premium_1' in feature_columns:
       features['is_premium_1'] = 1 if brand.lower() in ['bmw', 'mercedes-benz'] else 0
   
   # Popüler renk kontrolü
   if 'is_popular_color_1' in feature_columns:
       popular_colors = ['white', 'black', 'silver', 'gray']
       features['is_popular_color_1'] = 1 if color.lower() in popular_colors else 0
   
   # Clean title score
   if 'clean_title_score_1' in feature_columns:
       features['clean_title_score_1'] = 1 if title_status == 'clean vehicle' else 0
   
   return features

# Ana panel - Kullanıcı girdileri
st.header('Araç Özelliklerini Giriniz')

# 3 sütunlu layout
col1, col2, col3 = st.columns(3)

# İlk sütun
with col1:
   brand = st.selectbox('Marka', ['ford', 'chevrolet', 'toyota', 'honda', 'bmw', 'nissan', 'dodge', 'mercedes-benz'])
   year = st.slider('Model Yılı', 2000, 2024, 2020)
   mileage = st.number_input('Kilometre', min_value=0, max_value=300000, value=50000, step=1000)

# İkinci sütun
with col2:
   color = st.selectbox('Renk', ['white', 'black', 'silver', 'gray', 'blue', 'red'])
   state = st.selectbox('Eyalet', ['california', 'florida', 'texas', 'new york', 'pennsylvania'])

# Üçüncü sütun
with col3:
   title_status = st.selectbox('Araç Durumu', ['clean vehicle', 'salvage'])

# Tahmin butonu
if st.button('Fiyat Tahmini Yap', type='primary'):
   try:
       # Feature'ları hazırla
       input_features = prepare_features(brand, year, mileage, color, state, title_status)
       
       # Tahmin
       prediction = model.predict(input_features)[0]
       
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
