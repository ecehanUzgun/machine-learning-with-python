"""
 ctrl+ı : definition açılır.
"""
#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('satislar.csv')

# Veri Setini Bolme (Bagimsiz ve Bagimli Degiskenler)
aylar = veriler[['Aylar']] #aylar bagimsiz degisken
#test
print(aylar)

satislar = veriler[['Satislar']] #bagimli degisken
print(satislar)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
x_train_scaled = sc_X.fit_transform(x_train)
x_test_scaled = sc_X.transform(x_test)

# Bağımlı değişken için StandardScaler kullanmaya gerek yok!
# Çünkü Linear Regression ölçek bağımsız çalışabilir.

"""
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
"""

# model inşası (linear regression)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train) # Model eğitimde ölçeklenmemiş x_train ve y_train kullanmalı

"""
    X_train'den Y_train'i ogrendi.
    X_test'den kendi tahmin sonuclarını cikardi.
    Y_test herhangi bir sekilde sisteme dahil edilmedi.
"""

# Tahmin Yapma (Dikkat: X_test yerine x_test kullanılmalı!)
tahmin = lr.predict(x_test)

# Tahminleri Yazdır
print("Gerçek Değerler:\n", y_test.values)
print("Tahmin Edilen Değerler:\n", tahmin)

# Gerçek ve Tahmin Edilen Değerleri Grafik Üzerinde Gösterme
plt.scatter(x_test, y_test, color="red", label="Gerçek Değerler")  # Gerçek satışlar (Test seti)
plt.plot(x_test, tahmin, color="blue", label="Tahminler")  # Modelin tahmini
plt.xlabel("Aylar")
plt.ylabel("Satışlar")
plt.title("Gerçek vs Tahmin Edilen Satışlar")
plt.legend()
plt.show()