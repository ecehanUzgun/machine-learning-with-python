#1. Kutuphaneler
import numpy as np 
import pandas as pd 

#2. Veri Onisleme

#2.1 Veri Yukleme
veriler = pd.read_csv('eksikveriler.csv')

"""
    Eksik Veriler
    sci-kit learn
"""
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

"""
    encoder: Kategorik -> Numerik
"""
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

"""
    numpy dizileri dataframe donusumu 
"""
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
    dataframe birlestirme islemi
"""
# Ulke bilgileri ile Boy, Kilo ve Yas bilgilerini yatayda (axis=1) birlestiriyoruz.
s = pd.concat([sonuc,sonuc2], axis=1)
print(s)

# Daha once birlestirdigimiz verilere Cinsiyet bilgisini de ekliyoruz.
s2 = pd.concat([s,sonuc3], axis=1)
print(s2)

"""
    verilerin egitim ve test icin bolunmesi
"""
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0) #rastgele

"""
Verilerin Olceklenmesi (Feature Scaling)
    - Farkli olceklerdeki degiskenleri ayni araliga getirerek modelin daha dengeli ve dogru ogrenmesini saglayan bir on isleme adimidir.

fit_transform() metodu, x_train verisi uzerinde once fit (ogrenme) islemi yapar, ardindan transform (olcekleme) islemini uygular.
"""
from sklearn.preprocessing import StandardScaler 

sc = StandardScaler() #Ortalama (mean) degeri 0, standart sapması 1 olacak sekilde olcekleme yapar.

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

