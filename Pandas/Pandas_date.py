import numpy as np
import pandas as pd
from datetime import datetime

# Создаём набор отдельных компонент даты-времени
my_year = 2017
my_month = 1
my_day = 2
my_hour = 13
my_minute = 30
my_second = 15

# my_date = datetime(my_year, my_month, my_day)
# print(my_date)

# myser = pd.Series(['Nov 3, 2000', '2000-01-01', None])
# myserPd = pd.to_datetime(myser)
# print(myserPd)
# Если рассмотреть дату в варианте "01-03-2000", то
#  в Европе ее поймут как "день - месяц - год", но в США
# как "месяц - день - год"
#  тогда
# euro_date = '01-03-2000'
# print(pd.to_datetime(euro_date, dayfirst=True))
# strange_format_date = '22--Dec--2000'
# convert_strange = pd.to_datetime(strange_format_date,
#                                  format='%d--%b--%Y')
# print(convert_strange)

# _____________ Конвертация значений даты в формате строки в формат даты
# sales = pd.read_csv('./data//RetailSales_BeerWineLiquor.csv')
# sales['DATE'] = pd.to_datetime(sales['DATE'])
# print(sales['DATE'])
# и теперь мы можем применять соответсвующие методы
# print(sales['DATE'][0].year)
# ________ Можно при загрузке файла сразу произвести конвертацию в формат даты
sales = pd.read_csv('./data//RetailSales_BeerWineLiquor.csv', parse_dates=[0])
# ___ применение подобия группировки данных 
# sales = sales.set_index('DATE')
# print(sales.resample(rule='A').mean())
print(sales['DATE'].dt.year)
