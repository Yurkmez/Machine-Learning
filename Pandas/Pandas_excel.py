# При работе с таблицами макросы, формулы и визуализация игнорируются.
# Пандас работает с Excel-файлом как со словарем: # ключи - листы, значения -  датафреймы
# И необходимы доп. библиотеки:pip install openpyxl, xlrd

import pandas as pd
import numpy as np
# Исходный файл ./data/my_excel_file.xlsx
list_sheet_name = pd.ExcelFile('./data/my_excel_file.xlsx') # Узнать названия листов в файле
# print(list_sheet_name.keys())
# ____ Получить все данные из файла, а это СЛОВАРЬ! 
df_dict = pd.read_excel('./data/my_excel_file.xlsx', sheet_name=['First_Sheet', 'Second_Sheet'], engine='openpyxl')
# __ Запись в файл. Со СЛОВАРЕМ не получится записать несколько листов, только один лист в один файл:
df_dict['First_Sheet'].to_excel('./data/my_excel_file_single_sheet.xlsx', sheet_name='Первый лист')
# Используем метод "ExcelWriter" для записи нескольких листов:
with pd.ExcelWriter('./data/my_excel_file_multiple_sheet.xlsx', engine='openpyxl') as writer:  
    df_dict['First_Sheet'].to_excel(writer, sheet_name='Sheet_name_1')
    df_dict['Second_Sheet'].to_excel(writer, sheet_name='Sheet_name_2')