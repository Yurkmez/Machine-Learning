import numpy as np
import pandas as pd

df = pd.read_csv('./data/movie_scores.csv')
# print(df)
# print(df.isnull())
# print( df[ (df['first_name'].notnull()) &(df['pre_movie_score'].notnull())])
# __________удаление данных, содержащих NaN ________________________
# help(df.dropna)
# print(df.dropna())
# print(df.dropna(subset=['last_name']))

# __________ замена NaN данными __________________
# help(df.fillna)
# print(df.fillna("Yes!!!"))
# df['age'] = df['age'].fillna(100)
# __________________________ в конкретной колонке 
# print(df['age'].mean())
# print(df['age'].fillna(df['age'].mean()))
# __________________________ для всех числовых значений 
print(df.fillna(df.mean(numeric_only=True)))

