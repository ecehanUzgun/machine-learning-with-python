#1. Kutuphaneler
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

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

"""
Amaç: x_train ve y_train verilerini indekse göre sıralamak.
Neden? Eğer veri karışık bir sırada tutuluyorsa, doğru bir çizgi grafiği oluşturmak 
için indekslere göre sıralanması gerekir. Böylece aylar sırasıyla düzgün bir şekilde 
grafikte gösterilir.
"""
x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))

plt.title("Aylara Göre Satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")





def deneme(s):
    a = 1
    print(a)
    
    
     

     

    
deneme()    

    
    
    
    
    

