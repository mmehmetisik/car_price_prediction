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

# Feature hazırlama fonksiyonu
def prepare_features(brand, year, mileage, color, state, title_status):
   # Boş bir DataFrame oluştur
   features = pd.DataFrame(columns=feature_columns)
   
   # Temel özellikleri ekle
   features.loc[0, 'year'] = year
   features.loc[0, 'mileage'] = mileage
   features.loc[0, 'car_age'] = 2024 - year
   features.loc[0, 'avg_km_per_year'] = mileage / (2024 - year)
   features.loc[0, 'price_per_km'] = 0
   
   # Kategorik değişkenleri one-hot encode et
   features.loc[0, f'brand_{brand.lower()}'] = 1
   features.loc[0, f'color_{color.lower()}'] = 1
   features.loc[0, f'state_{state.lower()}'] = 1
   features.loc[0, f'title_status_{title_status.lower()}'] = 1
   
   # Premium marka kontrolü
   features.loc[0, 'is_premium_1'] = 1 if brand.lower() in ['bmw', 'mercedes-benz'] else 0
   
   # Popüler renk kontrolü
   popular_colors = ['white', 'black', 'silver', 'gray']
   features.loc[0, 'is_popular_color_1'] = 1 if color.lower() in popular_colors else 0
   
   # Clean title score
   features.loc[0, 'clean_title_score_1'] = 1 if 'clean' in title_status.lower() else 0
   
   # NaN değerleri 0 ile doldur
   features = features.fillna(0)
   
   return features

# Ana panel - Kullanıcı girdileri
st.header('Araç Özelliklerini Giriniz')

# 3 sütunlu layout
col1, col2, col3 = st.columns(3)

# İlk sütun
with col1:
   brand = st.selectbox('Marka', ['Ford', 'Chevrolet', 'Toyota', 'Honda', 'BMW', 'Nissan', 'Dodge', 'Mercedes-Benz'])
   year = st.slider('Model Yılı', 2000, 2024, 2020)
   mileage = st.number_input('Kilometre', min_value=0, max_value=300000, value=50000, step=1000)

# İkinci sütun
with col2:
   color = st.selectbox('Renk', ['White', 'Black', 'Silver', 'Gray', 'Blue', 'Red'])
   state = st.selectbox('Eyalet', ['California', 'Florida', 'Texas', 'New York', 'Pennsylvania'])

# Üçüncü sütun
with col3:
   title_status = st.selectbox('Araç Durumu', ['Clean Vehicle', 'Salvage'])

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
           st.write(f"- Premium Marka: {'Evet' if brand in ['BMW', 'Mercedes-Benz'] else 'Hayır'}")
           
       with col2:
           st.write(f"- Durum: {title_status}")
           st.write(f"- Lokasyon: {state}")
           st.write(f"- Renk: {color}")
           
   except Exception as e:
       st.error(f"Bir hata oluştu: {str(e)}")
       st.error("Lütfen tüm alanları doğru şekilde doldurunuz.")
