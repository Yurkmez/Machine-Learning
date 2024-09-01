import pandas as pd
import numpy as np
import os
# print(os.getcwd())
# Предполагается, что первая строка параметров - это заголовки
df = pd.read_csv('./data/example.csv')
# Если этого нам не надо, то: df = pd.read_csv('./data/example.csv', header=None)
# Если хотим назначить индекс (наприер, данные 1-й колонки) то:
# df = pd.read_csv('./data/example.csv', index_col=0)
# запись
print(df)
df.to_csv('./data/newExample.csv', index=False)
df_new = pd.read_csv('./data/newExample.csv')
print(df_new)
