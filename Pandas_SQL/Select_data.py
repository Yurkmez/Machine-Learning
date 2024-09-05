import pandas as pd
import numpy as np
import os

import datetime

import mysql.connector

# https://dev.mysql.com/doc/connector-python/en/connector-python-tutorial-cursorbuffered.html
# Данная ссылка демонстрирует более счложную работу с запросами и обновлением данных
# ___________________________________
config = {
        'user': 'root',
        'password': 'Ledzeppelin_7777',
        'host': '127.0.0.1',
        'database': 'employees',
    }
# __________________________

cnx = mysql.connector.connect(**config)
cursor = cnx.cursor()

query = ("SELECT first_name, last_name, hire_date FROM employees "
         "WHERE hire_date BETWEEN %s AND %s")

hire_start = datetime.date(1993, 1, 1)

hire_end = datetime.date(1999, 12, 31)

cursor.execute(query, (hire_start, hire_end))
# Здесь мы получаем список из кортежей, в каждом кортеже записи, т.е. [(...), (...), ...(...)] 
df_source = cursor.fetchall()

cursor.close()
cnx.close()

# print(df_source)

df = pd.DataFrame({"hire_start": df_source}) # трансформируем в датафрэйм
# print(df.iloc[0]) # выбираем отдельный кортеж 
# print(df.iloc[0].str[0]) # выбираем первое значение в выбранном кортеже

df.to_csv('data/resalt_select.csv', index=True)
df_new = pd.read_csv('data/resalt_select.csv')
print(df_new)




# for (first_name, last_name, hire_date) in cursor:
#   print("{}, {} was hired on {:%d %b %Y}".format(
#     last_name, first_name, hire_date))

