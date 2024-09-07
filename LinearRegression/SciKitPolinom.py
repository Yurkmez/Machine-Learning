import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error


df = pd.read_csv('data/Advertising.csv')
X = df.drop('sales', axis=1)
y = df['sales']

polinomial_converted = PolynomialFeatures(degree=2, include_bias = False)

# polinomial_converted.fit(X)                         # ____ Подготовка признаков для модели
# poly_features = polinomial_converted.transform(X)   # ____ Создание их
# ____ Это эквивалентно
poly_features = polinomial_converted.fit_transform(X)

# print(poly_features.shape)
# print(X.iloc[0])          # ___ Было 3 признака,
# print(poly_features[0])   # ___ теперь 9: 3 - исходных, 3 - их попарные перемножения, 3 - квадраты исходных (degree=2)

# ____ ! X contain 9 features ! ___________
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)
modelPR = LinearRegression()
modelPR.fit(X_train, y_train)
test_predictions = modelPR.predict(X_test)
MAE = mean_absolute_error(y_test, test_predictions)
MSE = mean_squared_error(y_test, test_predictions)
RMSE = np.sqrt(MSE)

# print("MAE: ", MAE)    # При линейной апроксимации - 1.213
# print("RMSE: ", RMSE)  # При линейной апроксимации - 1.516
# print("MSE: ", MSE)
print(modelPR.coef_)



