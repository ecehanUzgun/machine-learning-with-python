import numpy as np 
import pandas as pd 

"""
    DataFrame Nedir?
    Pandas kütüphanesinde kullanılan temel veri yapılarından biridir.
    Excel tablosuna benzer bir yapıya sahiptir ve satır-sütun formatında 
    verileri tutar.
"""

# Veri yukleme: 'eksikveriler.csv' dosyasını okuyarak bir DataFrame olusturuyoruz.
veriler = pd.read_csv('eksikveriler.csv')

# Eksik verileri doldurmak icin SimpleImputer sınıfını import ediyoruz.
from sklearn.impute import SimpleImputer 
# Eksik degerleri sutunun ortalami ile dolduracak
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# 'boy', 'kilo' ve 'yas' sütunlarını alıyoruz. Bu sütunları NumPy dizisine ceviriyoruz.
Yas = veriler.iloc[:,1:4].values #NumPy array
print(Yas)

# fit() fonksiyonu, eksik veri iceren sutunlardaki ortalama degerleri hesaplar.
imputer = imputer.fit(Yas[:,1:4])

# transform() eksik verileri, hesaplanan ortalamalarla doldurur.
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)

# 'ulke' sutunu bagimsiz bir NumPy dizisine cevrildi.
ulke = veriler.iloc[:,0:1].values
print(ulke)
    
# LabelEncoder(), Kategorik verileri sayisal degerlere cevirir
from sklearn import preprocessing
le = preprocessing.LabelEncoder() 

# 'ulke' sutunu icerisindeki kategorik degerleri sayisal degerlere donusturur
ulke[:,0] = le.fit_transform(veriler.iloc[:,0]) 
print(ulke)

#One-Hot Encoding, her kategoriyi ayri bir sutuna cevirir.
ohe = preprocessing.OneHotEncoder() 
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

print(list(range(22)))

#Ulke
sonuc = pd.DataFrame(data=ulke, index = range(22), columns = ['fr','tr','us'])
print(sonuc)

#Boy, Kilo ve Yas
sonuc2 = pd.DataFrame(data=Yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

#Cinsiyet
sonuc3 = pd.DataFrame(data = cinsiyet, index = range(22), columns = ['cinsiyet'])
print(sonuc3)

"""
    axis: Pandas'ta verileri birlestirme islemlerinde kullanilir.
    axis=0 - Satir bazinda birlestirme (alt alta ekler).
    axis=1 - Sutun bazinda birlestirme (yan yana ekler).
"""

# Ulke bilgileri ile Boy, Kilo ve Yas bilgilerini yatayda (axis=1) birlestiriyoruz.
s = pd.concat([sonuc,sonuc2], axis=1)
print(s)

# Daha once birlestirdigimiz verilere Cinsiyet bilgisini de ekliyoruz.
s2 = pd.concat([s,sonuc3], axis=1)
print(s2)





