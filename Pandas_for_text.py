import pandas as pd
import numpy as np
names = pd.Series(['andrew','bobo','claire','david','4'])
# print(names)
# преобразование серии в дата фрэйм
# tech_finance = ['GOOG,APPL,AMZN','JPM,BAC,GS']
# tickers = pd.Series(tech_finance)
# print(tickers)
# tickersDf = tickers.str.split(',', expand=True)
# print(tickersDf)
# ___________________ Исправление ошибо кв данных______________________
messy_names = pd.Series(["andrew  ","bo;bo","  claire  "])

def clearData(item):
    item = item.str.replace(';', '')
    item = item.str.strip()
    item = item.str.capitalize()
    return item

aaa = clearData(messy_names)
print(aaa)
