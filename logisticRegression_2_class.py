import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./data/hearing_test.csv')

# _________ Анализ данных __________________
# print(df.head())
# print(df['test_result'].value_counts())
# sns.countplot(data=df, x='test_result')
# plt.figure(dpi=100)
# sns.boxplot(x='test_result', y='age', data=df)
# sns.scatterplot(x='age', y='physical_score', data=df, hue='test_result', alpha=0.5)
# sns.pairplot(df, hue='test_result')
# sns.heatmap(df.corr(), annot=True)

# Строим 3-х мерный график
# fig=plt.figure()
# ax=fig.add_subplot(projection='3d')
# ax.scatter(df['age'], df['physical_score'], df['test_result'], c=df['test_result']) # ax.scatter(xs, ys, zs)

# __________ Обучение модели __________________
# ___ Разбиение на призначи и целевую переменную ______
X = df.drop('test_result', axis=1)
y = df['test_result']

# __ Разбиение данных 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
# __ Масштабирование признаков
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)
# __ Создем модель логистической регрессии
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
# __ Обучаем модель
log_model.fit(scaled_X_train, y_train)
# print(log_model.coef_) # Смотрим на коэффициенты
# __ Предсказываем
y_pred = log_model.predict(scaled_X_test)
# print(y_pred)

# __ Кроме самой предсказанной переменной, можно посмотреть на вероятность
# __ Это "pred_logo" - для логарифма вероятности и 
# __ "predict_proba" - для обычной вероятности (сокращение от probability - вероятность)
y_pred_proba = log_model.predict_proba(scaled_X_test)
# __ Вывод 2-х столбцов данных: 1- вероятность принадлежности к классу 1, 2 - к классу 2.
# print(y_pred_proba)

#  _____________ Метрики классификации: 
#  ______ Confusion Matrix и Accuracy, Precision, Recall и F1-Score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,ConfusionMatrixDisplay

# __ Оцениваем Accuracy
accuracy_resalt = accuracy_score(y_test, y_pred)
# print(accuracy_resalt)
# __ Оцениваем Precision
# __________ вначале строим матрицу accuracy
confusion_resalt = confusion_matrix(y_test, y_pred)
# print(confusion_resalt)
# ConfusionMatrixDisplay.from_estimator(log_model, scaled_X_test, y_test)
# ConfusionMatrixDisplay.from_estimator(log_model, scaled_X_test, y_test, normalize="all")
# ConfusionMatrixDisplay.from_estimator(log_model, scaled_X_test, y_test, normalize="true")

classification_resalt = classification_report(y_test, y_pred)
# print(classification_resalt)

# Можно отдельно использовать 
from sklearn.metrics import precision_score, recall_score

precision_score_resalt = precision_score(y_test, y_pred)
recall_score_resalt = recall_score(y_test, y_pred)
# print(precision_score_resalt, recall_score_resalt)

# PrecisionRecallDisplay.from_estimator и PrecisionRecallDisplay.from_predictions
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
# fig, ax = plt.subplots(figsize=(12,8))
# RocCurveDisplay.from_estimator(log_model, scaled_X_test, y_test, ax = ax)
# PrecisionRecallDisplay.from_estimator(log_model, scaled_X_test, y_test, ax=ax)

# Оцениваем вероятность отнесения к определенному классу
# print(log_model.predict_proba(scaled_X_test)[0], y_test[0])
# Получаем результат отнесения к определенному классу
print(log_model.predict(scaled_X_test)[0])

plt.show()