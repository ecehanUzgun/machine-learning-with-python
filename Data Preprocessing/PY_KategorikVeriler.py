"""
    Veri
        -Kategorik Veri
            -Nominal (siralanamaz ve olculemez. orn: ulkeler, renkler)
            -Ordinal (siralanabilir ama aralarindaki fark olculemez. orn: kucuk-orta-buyuk)
        -Sayisal Veri
            -Oransal (Ratio): 
             Gercek sifir noktası olan ve oran karsilastirmasi yapilabilen veriler.
             Orn: agirlik, uzunluk
            -Aralik (Interval)
             Sifir noktası keyfi olup sadece farklar anlamlidir.
             Orn: sicaklik, tarih
"""
import numpy as np #sayisal hesaplamalar
import pandas as pd #veri isleme , DataFrame

#veri yukleme
veriler = pd.read_csv('eksikveriler.csv')

# iloc: integer location, bütün satıların alınması için [:,]
# iloc[:, 0:1]  Tum satirlari alir (:) ve ilk sutunu secer (0:1).
# .values  Veriyi NumPy dizisine çevirir.
# ulke değişkeni şu an NumPy dizisi olarak ülke isimlerini içerir.
ulke = veriler.iloc[:,0:1].values
print(ulke)
    
from sklearn import preprocessing
le = preprocessing.LabelEncoder() #kategorik verileri sayisal degerlere cevirir.
#ornegin "Turkiye","Almanya","Fransa" gibi ulkeleri 0,1,2 gibi etiketlere cevirir.

ulke[:,0] = le.fit_transform(veriler.iloc[:,0]) #ilk sutundaki ulke verisini sayisal degerlere cevirir.
print(ulke)

ohe = preprocessing.OneHotEncoder() #One-Hot Encoding, her kategoriyi ayri bir sutuna cevirir.
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

