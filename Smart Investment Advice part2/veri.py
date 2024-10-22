import evds as e
import pandas as pd
import matplotlib.pyplot as plt
Apikey="D0u5VdZk8s"
evds=e.evdsAPI("your_key")


doviz=evds.get_data(["TP.DK.USD.A.YTL","TP.DK.EUR.A.YTL","TP.DK.GBP.A.YTL"],startdate="02-01-2005",enddate="21-10-2024",frequency=2)

gold=evds.get_data(["TP.MK.KUL.YTL"],startdate="02-01-2005",enddate="21-10-2024",frequency=2)

doviz.columns=["Tarih","USD","EUR","GBP"]
gold.columns=["Tarih","Gold"]


doviz.to_csv("doviz_kuru.csv")
gold.to_csv("altin.csv")


#altin grafik
plt.plot(gold['Tarih'], gold['Gold'], color='blue', label='Real')
plt.xlabel('tarih')
plt.ylabel('Altın')
plt.title('Altın Model')
plt.xticks(gold['Tarih'][::12], rotation=45)  # Her 100. tarihi göster ve 45 derece döndür
plt.tight_layout()  # Grafiği daha sıkı yerleştirir

# Y ekseni ölçeklendirme (örneğin logaritmik olarak)
  # Daha dengeli bir görünüm için log ölçek

# Legend'ı dışarı almak ve ortalamak
plt.legend(loc='upper right')  # Legend’ı sağ üst köşeye yerleştir

plt.show()

#dolar grafik

# Grafik boyutlarını ve düzeni ayarlama
fig = plt.figure(figsize=(10, 10))
rows = 3
columns = 1

# 1. Grafik - USD
ax1 = fig.add_subplot(rows, columns, 1)
ax1.plot(doviz['Tarih'], doviz['USD'], color='blue', label='USD')
ax1.set_xlabel('Tarih')
ax1.set_ylabel('USD')
ax1.set_title('USD Model')
ax1.set_xticks(doviz['Tarih'].index[::365])  # Her 365. günü göster
ax1.set_xticklabels(doviz['Tarih'][::365], rotation=45)
ax1.legend()

# 2. Grafik - EUR
ax2 = fig.add_subplot(rows, columns, 2)
ax2.plot(doviz['Tarih'], doviz['EUR'], color='green', label='EUR')
ax2.set_xlabel('Tarih')
ax2.set_ylabel('EUR')
ax2.set_title('EUR Model')
ax2.set_xticks(doviz['Tarih'].index[::365])
ax2.set_xticklabels(doviz['Tarih'][::365], rotation=45)
ax2.legend()

# 3. Grafik - GBP (İstersen ekleyebilirsin)
ax3 = fig.add_subplot(rows, columns, 3)
ax3.plot(doviz['Tarih'], doviz['GBP'], color='red', label='GBP')
ax3.set_xlabel('Tarih')
ax3.set_ylabel('GBP')
ax3.set_title('GBP Model')
ax3.set_xticks(doviz['Tarih'].index[::365])
ax3.set_xticklabels(doviz['Tarih'][::365], rotation=45)
ax3.legend()

# Grafikleri daha sıkı yerleştirir
plt.tight_layout()

# Tüm grafikleri göster
plt.show()
