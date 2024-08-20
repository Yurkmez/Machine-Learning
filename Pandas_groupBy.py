import pandas as pd
import numpy as np
df = pd.read_csv('./data/mpg.csv')
# print(df.describe())
# print(df.groupby('model_year').max())
# print(df.groupby('model_year').mean(numeric_only=True)['mpg'])
# print(df.groupby(['model_year', 'cylinders']).mean(numeric_only=True))
year_cyl = df.groupby(['model_year', 'cylinders']).mean(numeric_only=True)
# print(year_cyl)
# print(year_cyl.index.names)
# print(year_cyl.index.levels)
# print(year_cyl.loc[[70, 82]])
# print(year_cyl.loc[(70, 4)])
# print(year_cyl.xs(key=70, level='model_year'))
# print(year_cyl.xs(key=4, level='cylinders'))
# _________но xs не работает со списком (key не может иметь несколько значений)
#  поэтому, вначале фильтруем данные, например, оставляя только 
# к-во цилиндров = 4 и 8
# print(df[df['cylinders'].isin([4, 8])])
# print(df[df['cylinders'].isin([4, 8])].groupby(['model_year', 'cylinders']).mean(numeric_only=True))
# _________ сортировка ____________________
# print(year_cyl.sort_index(level='model_year', ascending=False))
# _______________Разные методы агрегации для различных колонок 
# print(df.agg(['mean','std'])['mpg'])
print(df.agg({'mpg': ['max','min'], 'weight': ['mean', 'std']}))