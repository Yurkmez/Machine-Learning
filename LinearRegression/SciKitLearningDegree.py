import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from joblib import dump, load


# Цикл для подбора модели оптимальной сложности
# 1. Создадим полиномиальные данные некоторой степени для данных X
# 2. Разобъём полиномиальные данные для обучающий и тестовый наборы данных
# 3. Выполним обучение модели на обучающем наборе данных
# 4. Посчитаем метрики на обучающем *и* тестовом наборе данных
# 5. Нанесём эти данные на график, чтобы увидеть момент переобучения модели


df = pd.read_csv('data/Advertising.csv')
X = df.drop('sales', axis=1)
y = df['sales']

# # ___Ошибка на обучающем наборе для той или иной степени полинома
# train_rmse_errors = []
# # ___Ошибка на тестовом наборе для той или иной степени полинома
# test_rmse_errors = []

# for d in range(1,10):
    
#     # ___Создаём полиномиальные данные для степени "d"
#     polynomial_converter = PolynomialFeatures(degree=d,include_bias=False)
#     poly_features = polynomial_converter.fit_transform(X)
    
#     # ___Разбиваем эти новые полиномиальные данные на обучающий и тестовый наборы данных
#     X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)
    
#     # ___Обучаем модель на этом новом наборе полиномиальных данных
#     model = LinearRegression(fit_intercept=True)
#     model.fit(X_train,y_train)
    
#     # ___Выполняем предсказание и на обучающем, и на тестовом наборе данных
#     train_pred = model.predict(X_train)
#     test_pred = model.predict(X_test)
    
#     # ___Вычисляем ошибки
    
#     # ___Ошибки на обучающем наборе данных
#     train_RMSE = np.sqrt(mean_squared_error(y_train,train_pred))
    
#     # ___Ошибки на тестовом наборе данных
#     test_RMSE = np.sqrt(mean_squared_error(y_test,test_pred))

#     # ___Добавляем ошибки в список для последующего нанесения на график
       
#     train_rmse_errors.append(train_RMSE)
#     test_rmse_errors.append(test_RMSE)
    
# plt.plot(range(1,6),train_rmse_errors[:5],label='TRAIN')
# plt.plot(range(1,6),test_rmse_errors[:5],label='TEST')
# plt.xlabel("Polynomial Complexity")
# plt.ylabel("RMSE")
# plt.legend()
# plt.show()

# Останавливаемся на модели со степенью "3"
final_polynomial_converter = PolynomialFeatures(degree=3,include_bias=False)
# print("1:", final_polynomial_converter)
final_poly_features = final_polynomial_converter.fit_transform(X)
# print("2:", final_poly_features.shape)

final_model = LinearRegression(fit_intercept=True)
final_model.fit(final_poly_features, y)

dump(final_model, './data/modelPR_final.joblib')
dump(final_polynomial_converter, './data/final_polynomial_converter.joblib')

loaded_converter = load('./data/final_polynomial_converter.joblib')
loaded_model = load('./data/modelPR_final.joblib')

compaiqn = [[149, 22, 12]]
transformed_data = loaded_converter.fit_transform(compaiqn)

resalt = loaded_model.predict(transformed_data)
print(resalt)