import pandas as pd

# Создаём DataFrame
data = {'Name': ['Anna', 'Ivan', 'Mariy'], 'Age': [25, 30, 22]}
df = pd.DataFrame(data)
# print(df)

# Выбор строки по метке индекса (loc)
str_1 = df.loc[0]

# Выбор строки по числовому индексу (iloc)
str_2 = df.iloc[0]

print(str_1)
print(str_2)
