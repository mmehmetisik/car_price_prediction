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

# Model yükleme
model_path = os.path.join('models', 'final_model.pkl')
with open(model_path, 'rb') as file:
   model = pickle.load(file)

# Ana başlık
st.title('🚗 Araç Fiyat Tahmin Uygulaması')
st.markdown("---")

# Yan panel (sidebar) için açıklama
st.sidebar.header("Uygulama Hakkında")
st.sidebar.markdown("""
Bu uygulama, araç özelliklerine göre fiyat tahmini yapar.
- Veri seti: USA Cars Dataset
- Model: LightGBM
- R² Score: 0.9856
- MAE: 844.21$
""")

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
   is_premium = 1 if brand in ['BMW', 'Mercedes-Benz'] else 0
   car_age = 2024 - year

# Tahmin butonu
if st.button('Fiyat Tahmini Yap', type='primary'):
   try:
       # Feature engineering (modelimizde kullandığımız aynı işlemleri yapacağız)
       input_data = pd.DataFrame({
           'brand': [brand],
           'year': [year],
           'title_status': [title_status],
           'mileage': [mileage],
           'color': [color],
           'state': [state],
           'car_age': [car_age],
           'is_premium': [is_premium]
       })
       
       # Tahmin
       prediction = model.predict(input_data)[0]
       
       # Sonuç gösterimi
       st.success(f'Tahmini Fiyat: ${prediction:,.2f}')
       
       # Detaylı açıklama
       st.markdown("---")
       st.markdown("### Fiyatı Etkileyen Faktörler")
       col1, col2 = st.columns(2)
       
       with col1:
           st.write(f"- Araç Yaşı: {car_age} yıl")
           st.write(f"- Kilometre: {mileage:,} km")
           st.write(f"- Premium Marka: {'Evet' if is_premium else 'Hayır'}")
           
       with col2:
           st.write(f"- Durum: {title_status}")
           st.write(f"- Lokasyon: {state}")
           st.write(f"- Renk: {color}")
           
   except Exception as e:
       st.error(f"Bir hata oluştu: {e}")
