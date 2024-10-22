import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Verilerinizi yükleyin
doviz_df = pd.read_csv('doviz_kuru.csv')  # CSV dosyanızın adını burada girin

# Verileri zaman serisi modeline uygun hale getirmek için tarih sırasına göre düzenleyelim
doviz_df['Tarih'] = pd.to_datetime(doviz_df['Tarih'], format='%d-%m-%Y')  # %d-%m-%Y formatı gün-ay-yıl şeklindedir
doviz_df = doviz_df.sort_values(by='Tarih')

# NaN değerlerini USD sütununda ortalama ile dolduralım
# Bu değerleri temizleyin
doviz_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Sonsuz değerleri NaN ile değiştir
doviz_df.dropna(inplace=True)  # Tüm NaN değerleri kaldır

# Sadece USD fiyatlarını alalım
doviz_values_usd = doviz_df['USD'].values.reshape(-1, 1)

# Verileri Min-Max ölçekleme ile normalleştirelim (LSTM için gerekli)
scaler_usd = MinMaxScaler(feature_range=(0, 1))
scaled_usd = scaler_usd.fit_transform(doviz_values_usd)

# Zaman serisi veri yapısı için bir yardımcı fonksiyon
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]  # X olarak
        X.append(a)
        Y.append(dataset[i + time_step, 0])  # Y olarak
    return np.array(X), np.array(Y)

# 60 adım (2 ay) geriye giderek veri setimizi hazırlayalım
time_step = 60

# USD için veri seti oluşturma
X_usd, Y_usd = create_dataset(scaled_usd, time_step)
X_usd = X_usd.reshape(X_usd.shape[0], X_usd.shape[1], 1)

# Eğitim ve test setlerini %80 - %20 oranında bölelim
def split_data(X, Y):
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]
    return X_train, X_test, Y_train, Y_test

# USD verilerini böl
X_train_usd, X_test_usd, Y_train_usd, Y_test_usd = split_data(X_usd, Y_usd)

# Modeli oluşturma fonksiyonu
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Modeli oluştur ve eğit
def train_model(X_train, Y_train, X_test, Y_test, scaler):
    model = create_lstm_model()
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, Y_train, batch_size=32, epochs=100, verbose=1, 
              validation_data=(X_test, Y_test), callbacks=[early_stop])
    
    # Tahmin yap
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Verileri orijinal ölçeklerine geri çevirelim
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    Y_test_inv = scaler.inverse_transform(Y_test.reshape(-1, 1))  # Y_test'i de orijinal haline getiriyoruz
    
    return train_predict, test_predict, Y_test_inv

# USD modeli eğit ve tahmin yap
train_predict_usd, test_predict_usd, Y_test_usd_inv = train_model(X_train_usd, Y_train_usd, X_test_usd, Y_test_usd, scaler_usd)

# Sonuçları görselleştirme
def plot_results(currency_name, real_data, train_predict, test_predict, time_step):
    plt.figure(figsize=(10, 6))

    # Boyutları düzeltme (flatten ile 2D'den 1D'ye indirgeme)
    train_predict = train_predict.flatten()
    test_predict = test_predict.flatten()
    
    # Eğitim tahminlerini çizelim (60 adım geriye gittiğimiz için başlangıcı kaydırıyoruz)
    plt.plot(doviz_df['Tarih'], real_data, label=f'Gerçek {currency_name} Fiyatları', color='blue')
    plt.plot(doviz_df['Tarih'][time_step:len(train_predict) + time_step], train_predict, label=f'Eğitim Tahminleri', color='green')
    
    # Test tahminlerini çizelim (eğitim tahminlerinin bittiği noktadan başlamalı)
    test_start_idx = len(train_predict) + (time_step)  # Test tahminlerinin başladığı yer
    test_end_idx = test_start_idx + len(test_predict)  # Test tahminlerinin bittiği yer
    plt.plot(doviz_df['Tarih'][test_start_idx:test_end_idx], test_predict, label=f'Test Tahminleri', color='red')

    # Grafik ayarları
    plt.xlabel('Tarih')
    plt.ylabel(f'{currency_name} Fiyatı')
    plt.title(f'{currency_name} Fiyatı LSTM Model Tahminleri')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# USD tahminlerini görselleştirme
plot_results('USD', doviz_df['USD'], train_predict_usd, test_predict_usd, time_step)

# MSE ve R2 Score hesaplama fonksiyonu
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

# USD için MSE ve R2 Score
mse_usd, r2_usd = calculate_metrics(Y_test_usd_inv, test_predict_usd)

# Sonuçları ekrana yazdıralım
print(f"USD için MSE: {mse_usd}")
print(f"USD için R² Score: {r2_usd}")
