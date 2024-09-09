import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/Advertising.csv')
# ____________________________________________________
# fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(10,4))

# axes[0].plot(df['TV'],df['sales'],'o')
# axes[0].set_ylabel("Sales")
# axes[0].set_title("TV Spend")

# axes[1].plot(df['radio'],df['sales'],'o')
# axes[1].set_title("Radio Spend")
# axes[1].set_ylabel("Sales")

# axes[2].plot(df['newspaper'],df['sales'],'o')
# axes[2].set_title("Newspaper Spend")
# axes[2].set_ylabel("Sales")
# plt.tight_layout()
# plt.show()
# Or __________________________________________________
# sns.pairplot(df, diag_kind='kde')
# plt.show()

# X, y _________________________
X = df.drop('sales', axis=1)
y = df['sales']

from sklearn.model_selection import train_test_split # _____ import !!!
# _____ random_state - обеспечивает повторяемость разбиения выборки. Зачем? При применении различных алгоритмов мы используем одинаковые наборы обучающей и тестовой выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression # _____ import !!!
modelLR = LinearRegression()
modelLR.fit(X_train, y_train)
resalt = modelLR.predict(X_test)
# __________ Ошибки модели ______________________________
from sklearn.metrics import mean_absolute_error,mean_squared_error # _____ import !!!
# print(df['sales'].mean())
# print(sns.histplot(data=df, x='sales', bins=20))
# plt.show()

# abs_error = mean_absolute_error(y_test, resalt)
# squared_error = mean_squared_error(y_test, resalt)
# print(abs_error)
# print(np.sqrt(squared_error))

# ______Анализ остатков __________________________
# test_residuals = y_test - resalt
# sns.scatterplot(x = y_test, y = test_residuals)
# plt.axhline(y=0, color='red', ls='--')
# sns.displot(test_residuals, bins=25, kde=True)
# plt.show()
# __________ Сравниваем с идеальным нормальным распределением
import scipy as sp # _____ import !!!
# Создаём объект figure и оси для рисования графика
# fig, ax = plt.subplots(figsize=(6,8),dpi=100)

# probplot возвращает значения, которые можно использовать при необходимости
# мы хотим просто нарисовать график, поэтому присваиваем эти значения переменной _
# _ = sp.stats.probplot(test_residuals,plot=ax)

# Итак, модель нас устраивает, обучаем ее на всех данных
modelLR_final = LinearRegression()
modelLR_final.fit(X, y)
# print(modelLR_final.coef_) # Коэффициенты, полученные на всей выборке
# print(modelLR_final.intercept_) # Коэффициент "0"

# print(modelLR.coef_) # Коэффициенты, полученные на тренировочной выборке
# Построим кое какие графики
# y_hat = modelLR_final.predict(X)
# _________________________________
# fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(10,4))

# axes[0].plot(df['TV'],df['sales'],'o')
# axes[0].plot(df['TV'],y_hat,'o',color='red')
# axes[0].set_ylabel("Sales")
# axes[0].set_title("TV Spend")

# axes[1].plot(df['radio'],df['sales'],'o')
# axes[1].plot(df['radio'],y_hat,'o',color='red')
# axes[1].set_title("Radio Spend")
# axes[1].set_ylabel("Sales")

# axes[2].plot(df['newspaper'],df['sales'],'o')
# axes[2].plot(df['newspaper'],y_hat,'o',color='red')
# axes[2].set_title("Newspaper Spend");
# axes[2].set_ylabel("Sales")
# plt.tight_layout();
# ________________________________________________
plt.show()
# Сохранение модели
from joblib import dump, load
# dump(modelLR_final, './data/modelLR_final.joblib')
# Загрузка (обученной) модели
load_model = load('./data/modelLR_final.joblib')
# print(load_model.coef_) 
# print(load_model.intercept_) 
# Загружаем в полученную модель данные по затратам на рекламу
campaign = [[149, 22, 12]]
resalt_load_model = load_model.predict(campaign)
print(resalt_load_model)