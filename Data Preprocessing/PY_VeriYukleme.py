# -*- coding: utf-8 -*-
"""
csv: Comma Separated Values (Virgülle Ayrılmış Değerler) anlamına gelir.
    - Bu dosya formatı, verileri satır ve sütun şeklinde saklayan düz metin dosyasıdır. 
    - Genellikle Excel veya veritabanı sistemlerinden veri aktarımı için kullanılır.

# veriler.csv ile PY_VeriYukleme.py aynı dizinde olmalı ya da dosya yolu yazılmalı
veriler = pd.read_csv('/users/ecehan/.../veriler.csv')
veriler = pd.read_csv('c:\\users\\ecehan\\...\\veriler.csv')
"""
    
#kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#veri yukleme
veriler = pd.read_csv('veriler.csv')
print(veriler)

#veri on isleme
boy = veriler[['boy']]
print(boy)

boykilo = veriler[['boy','kilo']]
print(boykilo)

x = 10