import numpy as np
import pandas as pd

df = pd.read_csv('./data/tips.csv')
# print(df)
# print(df.columns)
# print(df.index)
# print(df.head(10))
# print(df.tail(10))
# print(df.info())
# print(df.describe().transpose())
# print(df[['total_bill','tip']])
# Посчитаем % чаевых от суммы счета
procent = df['tip']/df['total_bill']*100
df['procent_tip'] = np.round(procent, 2)
# print(procent.mean())
# print(type(procent.mean()))
# df = df.set_index('Payment ID')
# print(df.loc["Sun2959"])
# ___________ Удаление строк ____________________
# df.drop('Sun2959', axis=0)
# По индексу так нельзя сделать!! -> df.drop(0, axis=0)
# Вариант
# df.iloc[1:] -> забираем все, кроме 0-го элемента
# ___________________ Добавлене строки (должна совпадать по к-ву колонок) ________________________________
# # one_row = df.iloc[0] # Допустим, это новая строка
# pd.concat(df, pd.DataFrame([one_row]), axis=0)
# print(df.head(10))
# print( df[ (df['total_bill']>40) & (df['tip']>5)] )
options = ['Sun', 'Sat']
print(df[df['day'].isin(options)])
