# ==========================
# 1. Gerekli Kütüphanelerin Eklenmesi
# ==========================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#pandas: Python'da veri analizi ve veri manipülasyonu 
#tablo benzeri veri yapıları (DataFrame ve Series) 

# ==========================
# 2. Veri Yükleme ve İnceleme
# ==========================
veriler = pd.read_csv('veriler.csv')
print(veriler)

# ==========================
# 3. Kategorik Verilerin Sayısal Hale Getirilmesi (Encoding)
# ==========================
#encoder: Kategorik -> Numeric
# Ülke sütunu kategorik veridir, önce label encoding sonra one-hot encoding yapılacak
ulke = veriler.iloc[:,0:1].values
print(ulke)
# Label Encoding (Ülkeleri sayisal degere cevirme)
from sklearn import preprocessing #preprocessing modül, veri ön işleme
le = preprocessing.LabelEncoder()
#[:,0] tüm satırları (:) ve ilk sütunu (0) temsil eder.
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ulke)
# One-Hot Encoding her ulke icin ayri sutun olusturulur.
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

# ==========================
# 4. Cinsiyet Verisini Encode Etme (Label Encoding ve One-Hot Encoding)
# ==========================
#encoder: Kategorik -> Numeric 
c = veriler.iloc[:,-1:].values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()   
c[:,-1] = le.fit_transform(veriler.iloc[:,-1])
print(c)
ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)

# ==========================
# 5. Pandas DataFrame Dönüşümleri
# ==========================
# Ülkeler için bir DataFrame oluşturuldu
sonuc = pd.DataFrame(data=ulke, index = range(22), columns = ['fr','tr','us'])
print(sonuc)
# TODO: Ulkeler icin dummy variable trap engellenecek!

# Boy, kilo ve yaş sütunlarını içeren DataFrame oluşturuldu
yas = veriler[['boy','kilo','yas']].values
sonuc2 = pd.DataFrame(data=yas, index = range(len(yas)), columns = ['boy','kilo','yas'])
print(sonuc2)

# Cinsiyet verisinin DataFrame dönüşümü
cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

#c'nin 0'dan 1'e kadar olan kısmı alinir.
cinsiyetS = pd.DataFrame(data = c[:,:1], index = range(22), columns = ['cinsiyet'])
print(cinsiyetS)

# ==========================
# 6. DataFrame Birleştirme İşlemleri
# ==========================
s=pd.concat([sonuc,sonuc2], axis=1) # Ülke ve yaş bilgileri birleştirildi
print(s)

#cinsiyet ile s birlestirildi, unutmamaliyiz ki cinsiyet sonraki asamalarda mutlaka sayisal olarak eklenmeli
s2=pd.concat([s,cinsiyetS], axis=1) # Cinsiyet bilgisi eklendi
print(s2)

"""
- train_test_split, scikit-learn kütüphanesinin model_selection modülünde bulunan bir fonksiyondur.
- Veri setini egitim (train) ve test setleri olarak boler.
- Makine ogrenimi modellerini egitmek ve test etmek icin kullanılır.
"""
#verilerin egitim ve test icin bolunmesi
#Bagimli degiskenler: ulke,boy,kilo,yas (Multiple Linear Regression)
#Bagimsiz degisken: cinsiyet
from sklearn.model_selection import train_test_split
x_trainC, x_testC,y_trainC,y_testC = train_test_split(s,cinsiyetS,test_size=0.33, random_state=0)

#verilerin olceklenmesi
"""
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_text = sc.fit_transform(x_test)
"""
# ==========================
# 8. Model Eğitimi (Linear Regression)
# ==========================
# 1. Gerekli kutuphane ice aktarilir
from sklearn.linear_model import LinearRegression
# 2. Linear regression modelinden nesne olusturulur.
regressor = LinearRegression()
# 3. Modeli egitiyoruz (fit)
#    X_train -> Bagimsiz degiskenler (girdi verileri)
#    Y_train -> Bagimli degisken (tahmin edilmek istenen deger)
regressor.fit(x_trainC,y_trainC)
# 4. Egitilmis model kullanilarak x_test verileri icin tahmin yapilir.
y_predC = regressor.predict(x_testC)

# ==========================
# 9. Boy Tahmini İçin Model Eğitimi
# ==========================
"""
    iloc[] fonksiyonu 
    iloc (Index Location), Pandas DataFrame'lerinde satır ve sütunları 
    indeksler ile seçmeye yarayan bir fonksiyondur.
    
    df.iloc[satır_indexleri, sütun_indexleri]
    boy = s2.iloc[:3].values      # İlk 3 satırı al
    boy = s2.iloc[::4].values     # Her 4 satırda bir al
    boy = s2.iloc[:3, 0].values   # İlk 3 satırın ilk sütununu al
"""
boy = s2.iloc[:,3].values # Boy sutunu secildi
print(boy)

sol = s2.iloc[:,:3] # Ilk 3 sutun (ulke bilgileri(one-hot encoding ile olusturulan))
sag = s2.iloc[:,4:] # Boy sutunu haric diger sutunlar
#Sol ve sag veriler birlestirildi, bagimsiz degiskenler
veri = pd.concat([sol,sag],axis=1) #sol ve sag birlestirildi
# Verilerin egitim ve test olarak bolunmesi
x_train, x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33, random_state=0)

r2 = LinearRegression()
r2.fit(x_train,y_train)
y_pred = r2.predict(x_test) # Boy tahmini yapıldı

"""
    TODO: Dummy Variable Trap engellenecek -
    Hata hesaplaması R^2 ve MSE ile hesaplanacak +
    Backward Elimination yapilacak +
"""
# ==========================
# 10. Backward Elimination İçin Model Hazırlığı
# ==========================
"""
- Çoklu doğrusal regresyon modeli oluşturmak için statsmodels kütüphanesi kullanıldı.
    boy Bağımlı değişken (tahmin edilmek istenen değişken)
    sm.OLS().fit() ile doğrusal regresyon modeli eğitiliyor.
    model.summary() ile modelin performansı ve değişkenlerin katsayıları inceleniyor
    
    Modelin özeti (summary):
    # R-squared: Modelin bağımlı değişkeni ne kadar iyi açıkladığını gösterir.
    # Bağımsız değişkenin bağımlı değişken üzerinde gerçekten etkisi var mı, yok mu?
        p-values: Değişkenin anlamlı olup olmadığını gösterir.
    # coefficients (beta katsayıları): Bağımsız değişkenlerin katsayıları
"""
import statsmodels.api as sm #istatistiksel modelleme yapmak için kullanılır.
#Ordinary Least Squares (OLS) yöntemi ile doğrusal regresyon 
#Sabit sutun ekleme
X = np.append(arr = np.ones((22,1)).astype(int), values=veri, axis=1)
"""
Amaç: Bağımsız değişkenler matrisine bir sabit sütun (bias/intercept) eklemek
Y = wX+b  -> Regresyon modelinin sabit terim içermesi sağlanır.
np.ones((22,1)).astype(int) → 22 satırlık 1 değerlerinden oluşan bir sütun (sabit terim için)
"""
#X_1 bağımsız değişkenler matrisini oluşturur
X_1 = veri.iloc[:,[0,1,2,3,4,5]].values #NumPy Dizisi
X_1 = np.array(X_1,dtype=float) #X_1 matrisi float türüne çevirir
model = sm.OLS(boy,X_1).fit() #Bagimli degisken boy, bagimsiz degisken X_1
print(model.summary()) #R-squared: 0.885

#.fit model verilerle eğitilir.

X_1 = veri.iloc[:,[0,1,2,3,5]].values 
X_1 = np.array(X_1,dtype=float) 
model = sm.OLS(boy,X_1).fit() 
print(model.summary()) #R-squared: 0.884

X_1 = veri.iloc[:,[0,1,2,3]].values 
X_1 = np.array(X_1,dtype=float) 
model = sm.OLS(boy,X_1).fit() 
print(model.summary()) #R-squared: 0.847
