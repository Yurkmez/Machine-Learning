import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV


from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.metrics import SCORERS
from joblib import dump, load

df = pd.read_csv('data/Advertising.csv')
X = df.drop('sales', axis=1)
y = df['sales']

polinomial_converted = PolynomialFeatures(degree=3, include_bias = False)

poly_features = polinomial_converted.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)

# ___ Масштабирование признаков

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaler = scaler.transform(X_train)
X_test_scaler = scaler.transform(X_test)
# print(X_train_scaler.mean())
# print(X_train_scaler.std())

# ____________________________________________________
# L1 - Ridge regression
# ____________________________________________________
# ___ Создание модели, обучение, оценка ошибок

# ridge_model = Ridge(alpha=10)
# ridge_model.fit(X_train_scaler, y_train)
# test_predictions = ridge_model.predict(X_test_scaler)

# ___ Ошибки на тестовом наборе данных

# MAE = mean_absolute_error(y_test,test_predictions)
# RMSE = np.sqrt(mean_squared_error(y_test,test_predictions))
# print(MAE, RMSE) # 0.577... 0.894...

# ___ Но насколько alpha=10 оптимальное решение?
# ___ Для этого RidgeCV - где производится перебор alpha и кросс-валидация

# print(SCORERS.keys())
# ridge_cv_model = RidgeCV(alphas=(0.1, 1.0, 10.0), scoring='neg_mean_absolute_error')
# ___ По сути, тестовый набор остается для валидации
# ___ т.к. для настройки гипер параметров мы используем только тренировочную выборку

# ridge_cv_model.fit(X_train_scaler, y_train)
# test_cv_predictions = ridge_cv_model.predict(X_test_scaler)

# ___ Какое альфа получилось лучшим?

# print(ridge_cv_model.alpha_)

# MAE_cv = mean_absolute_error(y_test,test_cv_predictions)
# RMSE_cv = np.sqrt(mean_squared_error(y_test,test_cv_predictions))
# print(MAE_cv, RMSE_cv) # 0.427... 0.618...
# print(ridge_cv_model.coef_)
# print(ridge_cv_model.best_score_)

# ____________________________________________________
# L2 - LASSO regression
# ____________________________________________________

# lasso_cv_model = LassoCV(eps=0.001, n_alphas=100, cv=5, max_iter=1000000)
# lasso_cv_model = LassoCV(eps=0.1, n_alphas=100, cv=5)
# lasso_cv_model.fit(X_train_scaler, y_train)
# print(lasso_cv_model.alpha_)
# test_lasso_cv_predictions = lasso_cv_model.predict(X_test_scaler)
# MAE_lasso_cv = mean_absolute_error(y_test,test_lasso_cv_predictions)
# RMSE_lasso_cv = np.sqrt(mean_squared_error(y_test,test_lasso_cv_predictions))
# print(MAE_lasso_cv, RMSE_lasso_cv) # 0.654... 1.130...

# ___ Результат хуже, но 

# print(lasso_cv_model.coef_)

# ____________________________________________________
# ElasticNetCV regression
# ____________________________________________________

elastic_cv_model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], eps=0.0011, n_alphas=100, max_iter=1000000)
elastic_cv_model.fit(X_train_scaler, y_train)
print(elastic_cv_model.l1_ratio_)
print(elastic_cv_model.alpha_)
test_elastic_cv_predictions = elastic_cv_model.predict(X_test_scaler)
MAE_elastic_cv = mean_absolute_error(y_test,test_elastic_cv_predictions)
RMSE_elastic_cv = np.sqrt(mean_squared_error(y_test,test_elastic_cv_predictions))
print(MAE_elastic_cv, RMSE_elastic_cv) # 0.654... 1.130...
print(elastic_cv_model.coef_)