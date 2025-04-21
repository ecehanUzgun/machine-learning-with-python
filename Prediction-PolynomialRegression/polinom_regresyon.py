#1. Kutuphaneler
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

#Veri Yukleme
veriler = pd.read_csv('maaslar.csv')

#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0) #rastgele

#veriler.iloc[satır_başlangıç:satır_bitiş, sütun_başlangıç:sütun_bitiş]
x = veriler.iloc[:,1:2] # 1. sütun dahil değil
y = veriler.iloc[:,2:]

# Linear Regression yaparsak nasıl olur?
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')