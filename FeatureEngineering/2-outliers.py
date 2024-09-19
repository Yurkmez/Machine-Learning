import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv('./data//Ames_Housing_Data.csv')
# Задача, предсказать цену продажи дома
# (1) посмотрим на целеваую переменную (ЦП) - SalePrice
# и на признаки, ктр сильно коррелируются с ЦП
df_corr_SalePrice = df.corr(numeric_only=True)['SalePrice'].sort_values()
# Видим, что Overall Qual имеет высокую корреляцию = 0.799262
# Построим график
# sns.scatterplot(x='Overall Qual', y='SalePrice', data=df)
# sns.scatterplot(x='Gr Liv Area', y='SalePrice', data=df)
# Смотрим на отсекаемые выбросы
# print(df[(df['Overall Qual']>8) & (df['SalePrice']<200000)])
# Выбираем 3 точки выбросов (глядя на график)
# print(df[(df['Gr Liv Area']>4000) & (df['SalePrice']<200000)])
# решаем, что их надо удалить
# Находим индексы
drop_ind = df[(df['Gr Liv Area']>4000) & (df['SalePrice']<200000)].index
df = df.drop(drop_ind, axis=0)
sns.scatterplot(x='Gr Liv Area', y='SalePrice', data=df)
# И так мы проходимся по остальным признакам, имеющим значимую корреляцию с ЦП
plt.show()

df.to_csv('./data/Outliers_removed.csv')