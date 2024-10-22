import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Verilerinizi yükleyin
altin_df = pd.read_csv('altin.csv')

# Verileri zaman serisi modeline uygun hale getirmek için tarih sırasına göre düzenleyelim
altin_df['Tarih'] = pd.to_datetime(altin_df['Tarih'], format='%Y-%m')
altin_df = altin_df.sort_values(by='Tarih')

# Sadece Altın fiyatı verisini alalım
altin_values = altin_df['Gold'].values.reshape(-1, 1)

# Verileri Min-Max ölçekleme ile normalleştirelim (LSTM için gerekli)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_altin = scaler.fit_transform(altin_values)

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
X, Y = create_dataset(scaled_altin, time_step)

# Veriyi LSTM için uygun hale getirmek üzere yeniden şekillendirelim
X = X.reshape(X.shape[0], X.shape[1], 1)

# Eğitim ve test setlerini %80 - %20 oranında bölelim
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size], X[train_size:len(X)]
Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

# LSTM modelini oluşturalım
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))  # %20 Dropout ekledik
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))  # %20 Dropout ekledik
model.add(Dense(25))
model.add(Dense(1))

# Modeli derleyelim
model.compile(optimizer='adam', loss='mean_squared_error')

# EarlyStopping callback ile overfitting'i önlemek
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Modeli eğitelim
history = model.fit(X_train, Y_train, batch_size=32, epochs=100, verbose=1, 
                    validation_data=(X_test, Y_test), callbacks=[early_stop])

# Tahminler yapalım
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Verileri orijinal ölçeklerine geri çevirelim
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Eğitim ve test sonuçlarını görselleştirme
plt.figure(figsize=(10, 6))

# Gerçek fiyatları çizelim
plt.plot(altin_df['Tarih'], altin_df['Gold'], label='Gerçek Altın Fiyatları', color='blue')

# Eğitim ve test tahminlerini ekleyelim
plt.plot(altin_df['Tarih'][time_step:train_size + time_step], train_predict, label='Eğitim Tahminleri', color='green')
plt.plot(altin_df['Tarih'][train_size + time_step + 1:], test_predict, label='Test Tahminleri', color='red')

# Grafik ayarları
plt.xlabel('Tarih')
plt.ylabel('Altın Fiyatı')
plt.title('Altın Fiyatı LSTM Model Tahminleri')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
