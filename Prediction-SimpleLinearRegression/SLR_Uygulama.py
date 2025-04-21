#1. Kutuphaneler
import numpy as np 
import pandas as pd 

"""
    X_train'den Y_train öğrendi.
    X_test'den de kendi tahmin sonuçlarını çıkardı.
    Y_test ve tahmin kıyaslanır.
"""

#2. Veri Onisleme

#2.1 Veri Yukleme
veriler = pd.read_csv('satislar.csv')
print(veriler)

#veri on isleme
aylar = veriler[['Aylar']] #aylar bagimsiz degisken
#test
print(aylar)

satislar = veriler[['Satislar']] #bagimli degisken
print(satislar)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0) #rastgele


#Verilerin Olceklenmesi (Feature Scaling)
#   - Farkli olceklerdeki degiskenleri ayni araliga getirerek modelin daha dengeli ve dogru ogrenmesini saglayan bir on isleme adimidir.
#fit_transform() metodu, x_train verisi uzerinde once fit (ogrenme) islemi yapar, ardindan transform (olcekleme) islemini uygular.
from sklearn.preprocessing import StandardScaler 

sc = StandardScaler() 
"""
# Yanlış Kullanım
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test) # Hata, test verisi yeniden ölçekleme 
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
#Test seti için ayrı bir ortalama ve standart sapma hesaplanıyor. 
#Eğitim ve test veri setleri için farklı ölçeklendirme kurallarına tabi tutuluyor.
#Model, gerçek dünyada kullanılamaz hale geliyor, çünkü test verisi farklı bir şekilde ölçeklenmiş oluyor.
"""
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.transform(y_test)

#model inşası (linear regression)
from sklearn.linear_model import LinearRegression
lr = LinearRegression() 
lr.fit(X_train,Y_train)

tahmin = lr.predict(X_test)
    










