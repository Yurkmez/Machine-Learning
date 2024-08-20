import numpy as np
import pandas as pd

data_one = {'A': ['A0', 'A1', 'A2', 'A3'],'B': ['B0', 'B1', 'B2', 'B3']}
data_two = {'C': ['C0', 'C1', 'C2', 'C3'], 'D': ['D0', 'D1', 'D2', 'D3']}

df_one = pd.DataFrame(data_one)
df_two = pd.DataFrame(data_two)

# print(pd.concat([df_one, df_two], axis=1))
# ______________ все таки хотим объединить по колонкам, тогда
# переименовываем колонки
# df_two.columns = df_one.columns
# затем объединяем по колонкам
# print(pd.concat([df_one, df_two], axis=0))
# но! тндексы повторяются, тогда
# df_new = pd.concat([df_one, df_two], axis=0)
#  and
# df_new.index = range(len(df_new))
# print(df_new)
#________________ Merge _____________________
registrations = pd.DataFrame({'reg_id':[1,2,3,4],'name':['Andrew','Bobo','Claire','David']})
logins = pd.DataFrame({'log_id':[1,2,3,4],'name':['Xavier','Andrew','Yolanda','Bobo']})
#________________ inner Merge _____________________

# innerData = pd.merge(registrations, logins, how='inner', on='name')
# print( innerData)
#________________ left & right Merge _____________________
# leftData = pd.merge(registrations, logins, how='left', on='name')
# rightData = pd.merge(registrations, logins, how='right', on='name')
# print(leftData)
# print(rightData)
#________________ outer Merge _____________________
# outerData = pd.merge(logins, registrations, how='outer', on='name')
# print(outerData)
# _______но, что если у нас в данных регистрации имена - индексы?
# registrations = registrations.set_index('name')
# print(registrations)
# _______как тогда объединить?
# someMerge = pd.merge(registrations, logins, left_index=True, right_on='name', how='inner')
# print(someMerge)
# _____ но, если названия колонок разные, перемименуем колонки в данных регистрацц
# registrations.columns = ['reg_id','reg_name', ]
# print(registrations)
# someMerge = pd.merge(registrations, logins, how = 'inner', left_on = 'reg_name', right_on = 'name').drop('reg_name', axis=1)
# print(someMerge)
# ____________ 
# print(registrations)
registrations.columns = ['id', 'name']
# print(registrations)
# print(logins)
logins.columns = ['id', 'name']
# print(logins)
someMerge = pd.merge(registrations, logins, how = 'inner', on = 'name', 
                     suffixes = ('_registration', '_login'))
print(someMerge)


