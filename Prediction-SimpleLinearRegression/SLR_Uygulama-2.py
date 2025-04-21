"""
Veri ölçeklendirme yapılmadan doğrusal regresyon modeli oluşturuldu.
"""
#1. Kutuphaneler
import numpy as np 
import pandas as pd 

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

#model inşası (linear regression)
from sklearn.linear_model import LinearRegression
lr = LinearRegression() 
lr.fit(x_train,y_train)

tahmin = lr.predict(x_test)
