############################################
# 1. Gerekli Kütüphaneleri Yükleme
############################################
# Temel Kütüphaneler
import numpy as np
import pandas as pd
import time
import warnings
import os
import pickle
import streamlit as st

# Veri Görselleştirme
import seaborn as sns
import matplotlib.pyplot as plt

# Scikit-learn preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

# Scikit-learn metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Temel Modeller
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Boosting Modelleri
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

##########################################
# 2. Veri Ön İşleme Ayarları
##########################################
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Görselleştirme ayarları
sns.set_theme()  # Seaborn temasını ayarla
plt.rcParams['figure.figsize'] = (10, 6)

# Warning mesajlarını kapatma
warnings.filterwarnings('ignore')

#########################################
# 3. Veri Setini Yükleme
#########################################

# CSV dosyasını okuma
df = pd.read_csv(r"C:\Users\ASUS\Desktop\car_price_project\USA_cars_datasets.csv")
df.head()

##############################
# Veriye ilk bakış
##############################

# Gereksiz kolonları atma
columns_to_drop = ['Unnamed: 0', 'lot', 'country', 'vin', 'condition']
df = df.drop(columns=columns_to_drop)
df.dtypes
df.head()

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head())
    print("##################### Tail #####################")
    print(dataframe.tail())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Numeric Summary #####################")
    numeric_cols = dataframe.select_dtypes(include=['int64', 'float64'])
    print(numeric_cols.describe().T)

check_df(df)

# 1. 0$ fiyatlı araçları kontrol edelim
print("##################### 0$ Fiyatlı Araçlar #####################")
print(df[df['price'] == 0][['price', 'brand', 'model', 'year', 'mileage']])
print(f"0$ fiyatlı araç sayısı: {len(df[df['price'] == 0])}")

# 2. 0 kilometreli araçları kontrol edelim
print("\n##################### 0 Kilometreli Araçlar #####################")
print(df[df['mileage'] == 0][['price', 'brand', 'model', 'year', 'mileage']])
print(f"0 kilometreli araç sayısı: {len(df[df['mileage'] == 0])}")

# 3. Aşırı yüksek kilometreli araçları kontrol edelim
# Üst sınır olarak 75% + 1.5*IQR kullanabiliriz (standart outlier tespiti)
Q1 = df['mileage'].quantile(0.25)
Q3 = df['mileage'].quantile(0.75)
IQR = Q3 - Q1
upper_limit = Q3 + 1.5 * IQR

print("\n##################### Aşırı Kilometreli Araçlar #####################")
print(df[df['mileage'] > upper_limit][['price', 'brand', 'model', 'year', 'mileage']].sort_values(by='mileage', ascending=False).head())
print(f"Aşırı kilometreli araç sayısı: {len(df[df['mileage'] > upper_limit])}")


# En yeni model yılını bulalım
max_year = df['year'].max()
# Yaklaşık 3 yıllık araçları "yeni" kabul edelim
new_car_threshold = max_year - 3

print("Veri setindeki en yeni yıl:", max_year)
print("3 yıllık araç eşiği:", new_car_threshold)

# Veri temizliği öncesi boyutu görelim
print("\nTemizlik öncesi veri boyutu:", df.shape)

# 1. 0 fiyatlı araçları çıkar
df = df[df['price'] != 0]

# 2. Mantıksız 0 km araçları çıkar (3 yıldan eski 0 km araçlar)
df = df[~((df['year'] < new_car_threshold) & (df['mileage'] == 0))]

# Temizlik sonrası boyutu görelim
print("\nTemizlik sonrası veri boyutu:", df.shape)

# Değişiklikleri kontrol edelim
print("\nKontroller:")
print("0$ fiyatlı araç sayısı:", len(df[df['price'] == 0]))
print("0 km araç sayısı:", len(df[df['mileage'] == 0]))

# Son durumda veri setinin özet istatistiklerini görelim
print("\nTemizlik sonrası özet istatistikler:")
print(df[['price', 'year', 'mileage']].describe())

#################
# Kategorik ve Nümerik Değişkenlerin Tespiti
#################

def grab_col_names(dataframe, cat_th=20, car_th=100):
    """

    Returns the names of categorical, numeric and categorical but cardinal variables in the data set.
    Note Categorical variables include categorical variables with numeric appearance.

    Parameters
    ------
        dataframe: dataframe
                Variable names of the dataframe to be taken
        cat_th: int, optional
                class threshold for numeric but categorical variables
        car_th: int, optinal
                class threshold for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numeric variable list
        cat_but_car: list
                List of cardinal variables with categorical appearance

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is inside cat_cols.
        The sum of the 3 return lists equals the total number of variables: cat_cols + num_cols + cat_but_car = number of variables

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car, num_but_cat

cat_cols, num_cols, cat_but_car,  num_but_cat = grab_col_names(df)

df.head(20)

cat_cols
num_cols
cat_but_car

###########################
# Kategorik Değişken Analizi
##############################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        'Ratio': 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print('##########################################')
    if plot:
        plt.figure(figsize=(12,6))
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col, plot=False)

#######################
# Nümerik Değişken Analizi
#######################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)

        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=False)

#########################################
# Analysis of Categorical Variables by Target
###########################################

def target_summary_with_cat(dataframe, target, categorical_col, plot=False):
    print(pd.DataFrame({'TARGET_MEAN': dataframe.groupby(categorical_col)[target].mean()}), end='\n\n\n')
    if plot:
        sns.barplot(x=categorical_col, y=target, data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    target_summary_with_cat(df, 'price', col, plot=False)

###################
# Analysis of Numeric Variables by Target
###################

def target_summary_with_num(dataframe, target, numerical_col, plot=False):
    print(pd.DataFrame({numerical_col+'_mean': dataframe.groupby(target)[numerical_col].mean()}), end='\n\n\n')
    if plot:
        sns.barplot(x=target, y=numerical_col, data=dataframe)
        plt.show(block=True)

for col in num_cols:
    target_summary_with_cat(df, 'price', col, plot=False)

################
# Analysis of Correlation
#################

def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    # Sadece numerik kolonları seç
    num_cols = [col for col in dataframe.columns if dataframe[col].dtype in ['int64', 'float64']]
    corr = dataframe[num_cols].corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]

    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="RdBu", annot=True, fmt=".2f")
        plt.title('Correlation Matrix of Numeric Variables')
        plt.show()

    return drop_list


high_correlated_cols(df, plot=True)

######################################
# Distribution of the Dependent Variable
######################################

df["price"].hist(bins=100)
plt.show(block=True)

##################################################
# Examining the Logarithm of the Dependent Variable
###################################################

np.log1p(df['price']).hist(bins=50)
plt.show(block=True)

###################
# Outliers Analysis
####################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

check_outlier(df, num_cols)

#######################
# Missing Value Analysis
#######################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)

##########################
# Rare Analysis
#########################

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ':', len(dataframe[col].value_counts()))
        print(pd.DataFrame({'COUNT': dataframe[col].value_counts(),
                            'RATIO': dataframe[col].value_counts() / len(dataframe),
                            'TARGET_MEAN': dataframe.groupby(col)[target].mean()}), end='\n\n\n')

rare_analyser(df, "price", cat_cols)


def rare_encoder_v2(dataframe, rare_perc_dict):
    temp_df = dataframe.copy()

    for col, rare_perc in rare_perc_dict.items():
        if col in temp_df.columns:
            tmp = temp_df[col].value_counts() / len(temp_df)
            rare_labels = tmp[tmp < rare_perc].index
            temp_df[col] = np.where(temp_df[col].isin(rare_labels), 'Rare', temp_df[col])

    return temp_df


# Her değişken için farklı eşikler belirleme
rare_perc_dict = {
    'brand': 0.005,  # %0.5
    'color': 0.02,  # %2
    'state': 0.01,  # %1
    'title_status': 0.05  # %5
}

df = rare_encoder_v2(df, rare_perc_dict)

df.head(20)

rare_analyser(df, "price", cat_cols)

###########################
# Feature Extraction
###########################

# 1. Araç Yaşı
df['car_age'] = 2024 - df['year']

# 2. Kilometre/Yaş Oranı (Yıllık ortalama kilometre)
df['avg_km_per_year'] = df['mileage'] / df['car_age']

# 3. Fiyat Segment Belirleme (Araç değer segmenti)
df['price_segment'] = pd.qcut(df['price'], q=5, labels=['very_cheap', 'cheap', 'medium', 'expensive', 'very_expensive'])

# 4. Kilometre Segmenti
df['mileage_segment'] = pd.qcut(df['mileage'], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])

# 5. Premium Marka Flag'i (Yüksek fiyatlı markaları işaretleme)
premium_brands = ['bmw', 'mercedes-benz', 'lexus', 'infiniti', 'maserati']
df['is_premium'] = df['brand'].isin(premium_brands).astype(int)

# 6. Popüler Renk Flag'i
popular_colors = ['white', 'black', 'silver', 'gray']
df['is_popular_color'] = df['color'].isin(popular_colors).astype(int)

# 7. Fiyat/Kilometre Oranı (Kilometreye göre değer)
df['price_per_km'] = df['price'] / (df['mileage'] + 1)  # +1 to avoid division by zero

# 8. Clean Title Score (Clean vehicle daha değerli)
df['clean_title_score'] = (df['title_status'] == 'clean vehicle').astype(int)

# Yeni oluşturulan feature'ları inceleyelim
print(df[['price', 'car_age', 'avg_km_per_year', 'price_segment', 'is_premium', 'price_per_km']].head())

df.head()

df.dtypes

cat_cols, num_cols, cat_but_car,  num_but_cat = grab_col_names(df)

cat_cols
num_cols
cat_but_car
num_but_cat

cat_cols = ['brand', 'title_status', 'color', 'state', 'price_segment',
            'mileage_segment', 'is_premium', 'is_popular_color', 'clean_title_score']

num_cols = ['price', 'year', 'car_age', 'mileage', 'avg_km_per_year', 'price_per_km']

num_but_cat = []

# Integer dönüşümleri
df['car_age'] = df['car_age'].astype('int64')
df['year'] = df['year'].astype('int64')

df[["year", "car_age"]].head()

# Kontrol edelim
print(df.dtypes)

######################
# Encoding
#####################

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()

####################
# Standardization Process
####################

num_cols = [col for col in num_cols if col not in ["price"]]

scaler = RobustScaler()

df[num_cols] = scaler.fit_transform(df[num_cols])

df.head(10)

######################
# Creating Model
######################

y = df["price"]

X = df.drop(["price", "model"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

print(X_train.dtypes)

###

# LightGBM modelini güncellenmiş parametrelerle tanımlama
lgbm_params = {
    'verbose': -1,  # Uyarıları kapatır
    'force_row_wise': True,  # Threading uyarısını kaldırır
    'feature_name': 'auto'  # Feature names uyarısını kaldırır
}

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('GBM', GradientBoostingRegressor()),
          ("LightGBM", LGBMRegressor(**lgbm_params)),
          ("CatBoost", CatBoostRegressor(verbose=False))]

rmse_scores = []
r2_scores = []
mae_scores = []
mse_scores = []
execution_times = []

for name, regressor in models:
    start_time = time.time()
    # Fit the model
    regressor.fit(X_train, y_train)
    # Make predictions
    y_pred = regressor.predict(X_test)
    # Calculate RMSE
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    rmse_scores.append(rmse)

    # Calculate R^2 score
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)
    # Calculate MAE
    mae = mean_absolute_error(y_test, y_pred)
    mae_scores.append(mae)
    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)
    # Calculate the execution time of the model
    execution_time = time.time() - start_time
    execution_times.append(execution_time)
    print(f"RMSE: {round(rmse, 4)} ({name})")
    print(f"R^2 Score: {round(r2, 4)} ({name})")
    print(f"MAE: {round(mae, 4)} ({name})")
    print(f"MSE: {round(mse, 4)} ({name})")
    print(f"Execution Time: {round(execution_time, 2)} seconds\n")

# En iyi modeli gösterme
best_model_idx = np.argmax(r2_scores)
print("\n" + "=" * 50)
print("\033[1mEn İyi Model:")
print(f"Model: {models[best_model_idx][0]}")
print(f"RMSE: {round(rmse_scores[best_model_idx], 4)}")
print(f"R^2 Score: {round(r2_scores[best_model_idx], 4)}")
print(f"MAE: {round(mae_scores[best_model_idx], 4)}")
print(f"MSE: {round(mse_scores[best_model_idx], 4)}")
print(f"Execution Time: {round(execution_times[best_model_idx], 2)} seconds\033[0m")
###

####################
# Hyperparameter optimization
####################

# Sadece CatBoost ve LightGBM için daha geniş parametreler
models = [
    ("CatBoost", CatBoostRegressor(verbose=False)),
    ("LightGBM", LGBMRegressor())
]

param_grids = {
    'CatBoost': {
        'iterations': [200, 300, 400],
        'learning_rate': [0.03, 0.05, 0.07],
        'depth': [5, 6, 7],
        'l2_leaf_reg': [3, 5],
        'border_count': [64, 96],
        'bagging_temperature': [1, 2]
    },
    'LightGBM': {
        'n_estimators': [200, 300, 400],
        'learning_rate': [0.03, 0.05, 0.07],
        'max_depth': [5, 6, 7],
        'num_leaves': [45, 63],
        'reg_lambda': [0.05, 0.1],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }
}

# Metrikleri saklamak için listeler
rmse_scores = []
r2_scores = []
mae_scores = []
mse_scores = []
execution_times = []
best_models = {}

# Modelleri eğit ve değerlendir
for name, regressor in models:
    print(f"\nHyperparameter Tuning for {name}:")
    print("-" * 50)

    start_time = time.time()

    # Grid Search
    grid_search = GridSearchCV(regressor,
                               param_grid=param_grids[name],
                               cv=5,
                               n_jobs=-1,
                               scoring='neg_mean_squared_error')

    grid_search.fit(X_train, y_train)

    # En iyi modeli sakla
    best_models[name] = grid_search.best_estimator_

    # Tahminler
    y_pred = grid_search.predict(X_test)

    # Metrikleri hesapla
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    execution_time = time.time() - start_time

    # Metrikleri listelere ekle
    rmse_scores.append(rmse)
    r2_scores.append(r2)
    mae_scores.append(mae)
    mse_scores.append(mse)
    execution_times.append(execution_time)

    # Sonuçları yazdır
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"RMSE: {round(rmse, 4)}")
    print(f"R^2 Score: {round(r2, 4)}")
    print(f"MAE: {round(mae, 4)}")
    print(f"MSE: {round(mse, 4)}")
    print(f"Execution Time: {round(execution_time, 2)} seconds")

# En iyi modeli bul
best_model_idx = np.argmax(r2_scores)
print("\n" + "=" * 50)
print("\033[1mEn İyi Model:")
print(f"Model: {models[best_model_idx][0]}")
print(f"RMSE: {round(rmse_scores[best_model_idx], 4)}")
print(f"R^2 Score: {round(r2_scores[best_model_idx], 4)}")
print(f"MAE: {round(mae_scores[best_model_idx], 4)}")
print(f"MSE: {round(mse_scores[best_model_idx], 4)}")
print(f"Execution Time: {round(execution_times[best_model_idx], 2)} seconds\033[0m")

#####################################
# Final Model and Prediction
#####################################

# En iyi LightGBM modelini final model olarak belirleme
final_model = best_models['LightGBM']  # Son hyperparameter tuning'den gelen en iyi model

# Test seti üzerinde tahmin
y_pred = final_model.predict(X_test)

# Sonuçları DataFrame'e dönüştürme
results = pd.DataFrame({
   'True Price': y_test,
   'Predicted Price': y_pred,
   'Difference': y_test - y_pred,
   'Absolute Difference': abs(y_test - y_pred),
   'Percentage Error': abs((y_test - y_pred) / y_test) * 100
})

# Özet istatistikler
print("\nModel Performance Metrics:")
print("-" * 50)
print(f"Mean Absolute Error: ${results['Absolute Difference'].mean():,.2f}")
print(f"Mean Percentage Error: %{results['Percentage Error'].mean():.2f}")
print(f"Median Absolute Error: ${results['Absolute Difference'].median():,.2f}")
print(f"Median Percentage Error: %{results['Percentage Error'].median():.2f}")

# En büyük 5 hata
print("\nTop 5 Largest Prediction Errors:")
print("-" * 50)
print(results.nlargest(5, 'Absolute Difference'))

# En küçük 5 hata
print("\nTop 5 Most Accurate Predictions:")
print("-" * 50)
print(results.nsmallest(5, 'Absolute Difference'))

########################
# Modeli Kaydetme
########################

# Proje dizinini belirle
project_dir = r"C:\Users\ASUS\Desktop\car_price_project"

# Models klasörünü oluştur
models_dir = os.path.join(project_dir, 'models')
os.makedirs(models_dir, exist_ok=True)

# Model yolunu belirle
model_path = os.path.join(models_dir, 'final_model.pkl')

# Modeli kaydet
with open(model_path, 'wb') as file:
   pickle.dump(final_model, file)

print(f"Model başarıyla kaydedildi: {model_path}")


###########################
# Streamlit
###########################


# 📌 Modelin eğitiminde kullanılan X_train'in sütunlarını kaydedelim
feature_columns = list(X_train.columns)  # Modelin eğitildiği sütunlar

# 📌 models klasörü yoksa oluştur
# os.makedirs("C:/Users/ASUS/Desktop/car_price_project/models", exist_ok=True)

# 📌 feature_columns.pkl dosyasını oluştur ve kaydet
feature_columns_path = "C:/Users/ASUS/Desktop/car_price_project/models/feature_columns.pkl"

with open(feature_columns_path, "wb") as file:
    pickle.dump(feature_columns, file)

print(f"Feature columns başarıyla kaydedildi! Dosya yolu: {feature_columns_path}")





















