import numpy as np
import pandas as pd
df = pd.read_csv('./data/tips.csv')
# print(df.describe().transpose())
#  False - сортировка по убыванию
# print(df.sort_values('tip', ascending=True))
# print(df.sort_values(['tip', 'sex'], ascending=True))
# print(df['tip'])
# aaa = df['tip'].idxmax()
# print(df.iloc[aaa])
# print(df.corr(numeric_only=True))
# print(df['sex'].value_counts())
# print(df['day'].unique())
# print(df['day'].nunique()) # print(len(df['day'].unique()))
# print(df['day'].value_counts())
# print(df['sex'].replace(['Female', 'Male'], ['F', 'M']))
# createDict = {'Female': 'F', 'Male': 'M'}
# print(df['sex'].map(createDict))
# print(df['sex'].map({'Female': 'F', 'Male': 'M'}))
# simple_df = pd.DataFrame([1, 2, 2, 2,], ['a', 'b', 'c', 'd'])
# print(simple_df.duplicated())
# print(simple_df.drop_duplicates())
# print(df['total_bill'].between(10, 20, inclusive='left'))
# print(df[df['total_bill'].between(15, 17, inclusive='left')])
# Выборка по условию
# print(df.nsmallest(5, 'tip'))
# Получение случайных данных
# print(df.sample(5))
# print(df.sample(frac=0.02))


