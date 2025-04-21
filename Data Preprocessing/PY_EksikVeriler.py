"""
    1. CSV dosyasindan veri okunur.
    2. Eksik veri iceren sutunlar secilir (1:4 sutunlari).
    3. SimpleImputer ile eksik veriler ortalama ile doldurulur.
    4. Guncellenmis veriler ekrana yazdirilir.
"""

import numpy as np #sayisal islemler
import matplotlib.pyplot as plt #grafik cizimleri
import pandas as pd #veri okuma, isleme ve analiz etme

#veri yukleme
veriler = pd.read_csv('eksikveriler.csv')
print(veriler)

"""
    - Eksik verileri islemek icin kullanilan bir scikit-learn sinifidir.
    - missing_values=np.nan Eksik degerleri (NaN) belirler.
    - strategy='mean' Eksik verileri ortalama (mean) degeri ile doldurur.
"""
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

Yas = veriler.iloc[:,1:4].values #NumPy array
print(Yas)

# fit() eksik veri iceren sutunlardaki ortalama degerleri hesaplar.
imputer = imputer.fit(Yas[:,1:4])

# transform() eksik verileri, hesaplanan ortalamalarla doldurur.
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)





