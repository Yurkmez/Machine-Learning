# ________ Необходимы доп. библиотеки!!! __________
#  pip install lxml
# ________________________________________________
import pandas as pd

url = "https://en.wikipedia.org/wiki/World_population"
# Если данная оперция не сработает, то можно сохранить
# сайт у себя на компьютере 
# Правая кнопка мыши - > View page source
# Получаем исходный код и сохраняем
tables = pd.read_html(url)
# print(len(tables))
# table_1 = tables[0].transpose()
table_2 = tables[1]
# print(tables[1].columns)
# table_1.columns = ['Region', '2020, %', '2030, %', '2050, %']
print(table_2)
table_2.to_html('./data/example.html', index=False)