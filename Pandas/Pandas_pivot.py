import numpy as np
import pandas as pd

df = pd.read_csv('./data/Sales_Funnel_CRM.csv')
# print(df)
# _______? Сколько лицензий купила компания Гугл.
licenses = df[['Company', 'Product', 'Licenses']]
aboutCompany = pd.pivot(data=licenses, index='Company', columns='Product', values='Licenses')
# _______? Как найти сумарное к-во лицензий и сумму продаж для каждой компании
aboutSumCompany = pd.pivot_table(df, index='Company', aggfunc='sum', values=['Sale Price', 'Licenses'])
# То же
# aboutSumCompany_2 = df.groupby('Company').sum()
# print(aboutSumCompany_2)
# ____ ? Сколько менеджер ...
aboutSumManager = pd.pivot_table(df, index=['Account Manager', 'Contact'], aggfunc='sum', values='Sale Price')
print(aboutSumManager)
print('_______________________________')

aboutSumManager_2 = pd.pivot_table(df, index=['Account Manager', 'Contact'], aggfunc=[np.sum, np.mean], values='Sale Price')
print(aboutSumManager_2)
print('_______________________________')

aboutSumManager_3 = pd.pivot_table(df, index=['Account Manager', 'Contact'], aggfunc='sum', values='Sale Price', columns='Product')
print(aboutSumManager_3)
print('_______________________________')

aboutSumManager_4 = pd.pivot_table(df, index=['Account Manager', 'Contact', 'Product'], aggfunc='sum', values='Sale Price')
print(aboutSumManager_4)
print('_______________________________')

# параметр "margins" выводит общую сумму 
aboutSumManager_5 = pd.pivot_table(df, index=['Account Manager', 'Contact', 'Product'], aggfunc='sum', values='Sale Price', margins=True)
print(aboutSumManager_5)