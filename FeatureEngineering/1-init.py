import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_ages(mu=50,sigma=13,num_samples=100,seed=42):

    # Указываем значение random seed в той же ячейке, что и вызов метода random - 
    # это нужно для того, чтобы получить те же самые данные
    # Мы используем значение 42 (42 это число из комедийного сериала Автостопом по Галактике - 
    # Hitchhiker's Guide to the Galaxy)
    np.random.seed(seed)

    sample_ages = np.random.normal(loc=mu,scale=sigma,size=num_samples)
    sample_ages = np.round(sample_ages,decimals=0)
    
    return sample_ages

sample = create_ages()

# Изучаем данные
# sns.displot(sample, bins = 20)
# sns.boxplot(x=sample) # Одна точка вышла за пределы, как найти? 
ser = pd.Series(sample)

# print(ser.describe())
# count    100.00000
# mean      48.66000
# std       11.82039
# min       16.00000
# 25%       42.00000
# 50%       48.00000
# 75%       55.25000
# max       74.00000
# dtype: float64


# определяем интерквартильный интервал

IQR = 55.25000 - 42.00000
lower_limit = 42.00000 - 1.5 * IQR
# print(lower_limit)
# Исключаем точку выброса
# print(ser[ser > lower_limit])

# Получить точки процентилей можно программно
q75, q25 = np.percentile(sample, [75, 25])
print(q75, q25)


plt.show()
