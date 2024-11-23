# Какова логика работы с машинным обучением

## 1. Импорт библиотек

<font color="yellow">import numpy as np  
 import pandas as pd  
 import matplotlib.pyplot as plt  
 import seaborn as sns</font>

## 2. Перенос данных из источника в датафрейм

<font color="yellow">df = pd.read_csv("ххх.csv")</font>

## 3. Анализ данных

Кстати, неплохой ресурс по очистке данных https://habr.com/ru/articles/704608/

### <font color="lightgrey"><u>3.1. Общий анализ данных</font></u>

<font color="yellow">df.head()  
 df.info()  
 df.describe()  
 df.isnull().sum()  
df = df.dropna() <font color="lightgreen"> _- удаляем наблюдения если в них есть хотя бы 1 значение NAN_</font>  
 df['ааа'].unique() <font color="lightgreen"> _- смотрим, есть ли в колонке "ааа" какие-то не предусмотренные значения, например "-" или "."_</font>  
 df[['ааа','ссс']].value*counts()  
df[df['ааа'] == 'что-то'].groupby('ккк').describe().transpose() <font color="lightgreen"> *- смотрим параметры значений 'что-то' в колонке 'ааа', сгруппированных по колонке 'ккк'_</font>  
 df = df[df['ххх']!='.'] <font color="lightgreen"> _- если нас что то не устраивает в определенной колонке
-- то мы можем либо переписать датафрейм без данного значения\_</font>  
 df.at[kkk, 'ххх'] = 'что-то' <font color="lightgreen"> \*- либо заменить значение с индексом "kkk" в колонке "ххх" на другое\_</font></font>

### <font color="lightgrey"><u>3.2. Визуализация</font></u>

<font color="yellow">1 - sns.scatterplot(x='название колонки',y='название колонки',data=df,hue='целевая колонка',palette='Dark2')  
 2 - sns.pairplot(df,hue='целевая колонка',palette='Dark2')  
 3 - sns.catplot(x='название колонки',y='название колонки',data=df,kind='box',col='по какой колонке разбить, если надо',palette='Dark2')  
 4 - sns.hisplot(data=df, x='название колонки', bins=20)  
 5 - ... </font>

### <font color="lightgrey"><u>3.3. Oцифровывание категориальных признаков</font></u>

<font color="yellow">pd.get*dummies(df)  
<font color="lightgreen"> *- если не хотим оцифровывать целевую переменную, то:\_</font>
pd.get_dummies(df.drop('целевая колонка', axis = 1))  
<font color="lightgreen"> *- если какая-то колонка дублируется, то добавляем "drop*first=True":\*</font>
pd.get_dummies(df.drop('целевая колонка',axis=1),drop_first=True)</font>

## 4. Собственно машинное обучение (алгоритм действий)

### <font color="lightgrey"><u>4.1. Выбор модели</font></u>

<font color="yellow">from sklearn.linear_model import LinearRegression  
 myModel = LinearRegression(some params ...)</font>

### <font color="lightgrey"><u>4.2. Разбивка данных на обучающую и тестовую выборки</font></u>

<font color="yellow">X = df.drop('целевая колонка',axis=1)  
 y = df['целевая колонка']</font>

### Примечание

Если нам надо создать дополнительные признаки,
например, для полиномиальной модели регрессии, то
вначале генерируем модель создания признаков (см. детали в Приложении, про полиномиальную модель).

<font color="yellow">from sklearn.preprocessing import PolynomialFeatures  
polynomial_converter = PolynomialFeatures(<font color="orange">degree=2,include_bias=False</font>)</font>

т.е. **polynomial_converter** - это модель признаков где есть заданные параметры, то есть, когда мы в конечном итоге сгенерируем обученную модель и ее куда-либо передадим, то вместе с ней мы должны передать и модель генерации признаков !!!

<font color="yellow">poly_features =polynomial_converter.fit_transform(X)</font>

И уже эти признаки выполняют роль исходных (poly_features ~ X), которые мы подаем на обучающую модель.

И, собственно, разбиение данных:

<font color="yellow">from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(Х, y, test_size=0.3, random_state=101)</font>

#### <font color="lightgrey">Масштабирование данных (при необходимости)</font>

<font color="yellow">from sklearn.preprocessing import StandardScaler  
<font color="lightgreen">Это "стандартизация", то есть - по параметрам среднего и СКО, в отличии от нормализации - "мин - макс"</font>  
 scaler = StandardScaler()  
 scaler.fit(X_train)

#### <font color="pink">Важно! Метод fit здесь вычисляет параметры стандартизации - в данном случае среднее и СКО. Чтобы эти вычисляемые параметры не учитывали данные из тестового набора данных, они вычисляются только на тренировочных данных!!!</font>

X_train_scaler = scaler.transform(X_train)  
 X_test_scaler = scaler.transform(X_test)</font>

### <font color="lightgrey"><u>4.3. Провинутые методы разбивки данных и, как следствие, продвинутые методы обучения</font></u>

---

#### <font color="lightgrey"><u>4.3.1. Разбивка данных на "Train", "Validation", "Test"</font></u>

1. Обучение на данных "Train"
2. Проверка и подбор гиперпараметров на "Validation"
3. Финальная проверка на данных "Test"

#### Вызываем SPLIT дважды! Здесь мы создаём три набора данных - TRAIN, VALIDATION и TEST

Разбиение данных для Train и (Validation + Test):

<font color="yellow">from sklearn.model_selection import train_test_split

X_train, X_other, y_train, y_other = train_test_split(Х, y, test_size=0.3, random_state=101)

X_eval, X_test, y_eval, y_test = train_test_split(X_other, y_other, test_size=0.5, random_state=101)</font>

Каждый набор будет содержать по 15% данных от исходного набораю

#### <font color="pink">Порядок переменных в формулах важен!!!</font>

Можно проверить получившиеся объемы выборок  
<font color="yellow">len(df), len(X_train), len(X_eval), len(X_test)</font>

Масштабируем данные (SCALE)  
<font color="yellow">from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
X_eval = scaler.transform(X_eval)  
X_test = scaler.transform(X_test)</font>

Как видно, мы подвергаем процедуре **fit** (т. е. вычисляем среднее и СКО) только на тренировочных данных.

Далее обучаем, предсказываем на наборе X_eval, оптимизируем и проверяем на тестовом наборе точность предсказания.

---

#### <font color="lightgrey"><u>4.3.2. Кросс-валидация с помощью **cross_val_score**</font></u>

![image](grid_search_cross_validation.png)
Эта процедура <u>не предназначена для обучения</u>, а используется для оценки того, на сколько частей следует разбивать тренировочный набор данных.

Как обычно, анализируем, разбиваем (но, несмотря на присутствии 3-х наборов данных, мы разбиваем данные на 2 набора - тренировочный и тестовый), масштабируем.Далее

<font color="yellow">model = Ridge(alpha=100)

from sklearn.model_selection import cross_val_score

<font color="lightgreen">Мы можем, без обучения модели, оценить ее потенциал в зависимости от выбранных гиперпараметров.</font>

scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error',cv=5)</font>

https://scikit-learn.org/stable/modules/model_evaluation.html

cv = 5 - означает, что мы разбиваем обучающую выборку на 5 частей, пять раз на 4/5 тренировочной выборки обучаем модель, и оцениваем каждую по параметру 'neg_mean_squared_error'.

Посмотреть, что получилось  
<font color="yellow">print(scores)  
print(abs(scores.mean()))</font>

Ну и обучаем, предсказываем и оцениваем ее

<font color="yellow">model.fit(X_train,y_train)  
y_final_test_pred = model.predict(X_test)  
mean_squared_error(y_test,y_final_test_pred)</font>

---

#### <font color="lightgrey"><u>4.3.3. Кросс-валидация с помощью **cross_validate**</font></u>

Функция **cross_validate** отличается от **cross_val_score** двумя аспектами:

эта функция позволяет использовать для оценки несколько метрик;

она возвращает не только оценку на тестовом наборе (test score), но и словарь с замерами времени обучения и скоринга, а также - опционально - оценки на обучающем наборе и объекты estimator.

В случае одной метрики для оценки, когда параметр scoring является строкой string, вызываемым объектом callable или значением None, ключи словаря будут следующими:

        - ['test_score', 'fit_time', 'score_time']

А в случае нескольких метрик для оценки, возвращаемый словарь будет содержать следующие ключи:

    ['test_<scorer1_name>', 'test_<scorer2_name>', 'test_<scorer...>', 'fit_time', 'score_time']

return_train_score по умолчанию принимает значение False, чтобы сэкономить вычислительные ресурсы. Чтобы посчитать оценки на обучающем наборе, достаточно установить этот параметр в значение True.

<font color="yellow">from sklearn.model_selection import cross_validate

scores = cross_validate(model, X_train, y_train, scoring =['neg_mean_absolute_error', 'neg_mean_squared_error', 'max_error'], cv=5)</font>

https://scikit-learn.org/stable/modules/model_evaluation.html

Смотрим результата

<font color="yellow">pd.DataFrame(scores)  
pd.DataFrame(scores).mean()</font>

Здесь уже, по сравнению с **cross_validate_scores**
мы получаем гораздо более полную информацию: время оценки параметров и ошибок, сами ошибки

<font color="yellow">scores.mean()</font>

Ну и создаем новую модель (c новым гиперпараметром)  
<font color="yellow">model = Ridge(alpha=1)</font>

Снова оцениваем (например)  
<font color="yellow">scores.mean()</font>

Короче, находим наилучшую и уже на ней обучаем

<font color="yellow">model.fit(X_train,y_train)</font>  
И далее предикт, оценка на тестовом наборе, обучение на всем наборе, сохранение модели.

---

### <font color="lightgrey"><u>4.3.4. Обобщённый метод поиска по сетке - **grid search.**</font></u>

Этот класс: **GridSearchCV** - позоляет перебирать гиперпараметры (предаваемые как словарь) с применением кросс-валидации.

Поиск по сетке:

-   выбор функции оценки - estimator (regressor или classifier)
-   выбор пространства параметров;
-   метод поиска или сэмплирования кандидатов (со схемой кросс-валидации)
-   выбор функции оценки (score function).

---

<font color = 'lightgreen'>**estimator** - ' оценщик', собственно метод обучения.  
Если мы работаем в рамках линейных моделей, то

### _Классические линейные регрессоры:_

<u>LinearRegression</u> - обычная линейная регрессия методом наименьших квадратов;  
<u>Ridge</u> - линейный метод наименьших квадратов с регуляризацией l2;  
<u>RidgeCV</u> - гребневая регрессия со встроенной перекрестной проверкой;  
<u>SGDRegressor</u> - Линейная подгонка путём минимизации регуляризованных эмпирических потерь с помощью SGD.

### _Регрессоры с выбором переменных:_

эти методы оценки имеют встроенные процедуры подбора переменных, но метод оценки, использующий штраф L1 или эластичную сеть, также это делает (обычно это SGDRegressor или SGDClassifier).  
(примеры)  
<u>ElasticNet</u> - Линейная регрессия с комбинированными регуляризаторами L1 и L2.  
<u>ElasticNetCV</u> - Модель эластичной сети с итеративной подгонкой по пути регуляризации.
<u>Lasso</u> - Модель обучения с использованием L1-регуляризации (также известной как «Лассо»).  
<u>LassoCV</u> - Линейная модель Лассо с итеративной подгонкой по пути регуляризации.  
<u>и другие ...</u></font>

---

Выбираем модель оценки  
<font color='yellow'>from sklearn.linear_model import ElasticNet</font>  
Выбираем пространство параметров  
<font color='yellow'>base_elastic_model = ElasticNet()
param_grid = {'alfa': [0.1, 1, 5, 50, 100], 'l1_ratio':[.1, .5, .7, .9, .95, .99, 1]}</font>  
Выбираем метод поиска  
<font color='yellow'>from sklearn.model_selection import GridSearchCV

grid_model = GridSearchCV(estimator=base_elastic_model,
param_grid=param_grid,
scoring='neg_mean_squared_error',
cv=5,
verbose=2)</font>

<font color="lightgreen">Примечания. verbose - показывает некоторые параметры в процессе обучения</font>

<font color='yellow'>grid_model.fit(X_train,y_train)</font></font>

<font color='lightgreen'>Эта модель обучается, перебирая гиперпараметры для каждого набора разбитых на 5 частей данных в обучающей выборке, и найдет наилучшее сочетание параметров </font>

Смотрим, что получили

<font color='yellow'>grid*model.best_estimator*  
grid*model.best_params*  
pd.DataFrame(grid*model.cv_results*)</font>

Далее вычисляем ошибку  
<font color='yellow'>y_pred = grid_model.predict(X_test)  
from sklearn.metrics import mean_squared_error  
mean_squared_error(y_test,y_pred)</font>

Но есть и такой поиск по сетке, как случайный
RandomizedSearchCV,
для которого (например)
param_distributions = {"alfa": scipy.stats.uniform(0.1, 99),
'l1_ratio':scipy.stats.uniform(0.1, 0.9)}

### <font color="lightgrey"><u>4.4. Обучение</font></u>

<font color="yellow">myModel.fit(X_train_scaler, y_train)</font>

### <font color="lightgrey"><u>4.5. Предсказание</font></u>

<font color="yellow">y_predict = myModel.predict(X_test)</font>

### <font color="lightgrey"><u>4.6. Оценка качества</font></u>

<font color="yellow">from sklearn.metrics import mean_absolute_error,mean_squared_error  
 MAE = mean_absolute_error(y_test, y_predict)  
 MSE = mean_squared_error(y_test, y_predict)  
 RMSE = np.sqrt(MSE)

---

_<font color="lightgreen"> - анализ остатков (residuals), <u>актально для моделей линейной регрессии</u>: </font>_
_delta = y_test - y_predict_  
_sns.scatterplot(x=y_test, y = y_predict)_  
_plt.axhline(y=0, color='r', ls='--')_  
_<font color="lightgreen"> структура остатков должна быть достаточно случайная, не регулярная, а плотность распределения_</font>  
_sns.displot(delta, bins=25, kde=True)_  
_<font color="lightgreen">должна быть
близкой к нормальной. Можно посмотреть, насколько распределение остатков близко к ноормальному. Для этого:_</font>  
 _import scipy as sp_  
 _fig, ax = plt.subplots(figsize=(6, 8), dpi=100)_  
 _\_= sp.stats.probplot(delta, plot=ax)_  
<font color="lightgreen">_И здесь по диагонали будет прочерчена линия для расположения точек, распределенных по нормальному закону, и реальные точки. По их отклонению от линии можно судить, насколько распределение ошибок отлично от нормального закона._</font></font>

### <font color="lightgrey">4.7. Регуляризация</font>

---

### <u>Ridge Regression (Гребневая регрессия)</u>

Здесь мы вводим штраф за переобученность, минимизируя параметр "альфас умножить на сумму квадратов коэффициентов", соответственно, сила штрафа определяется регуляризацией коэффициента альфа (alpha). В принципе, это та же линейная модель, только с параметром регуляризации. Можно модель Ridge, но лучше RidgeCV (с кросс-валидацией). Кстати, alphas=(0.1, 1.0, 10.0) - параметры по умолчанию,
а поиск варианта метрики ошибки (Кстати, они все со знаком минус, поэтому, чем больше их в-на, тем лучше):  
<font color="yellow">from sklearn.metrics import SCORERS  
SCORERS.keys()</font>

<font color="yellow">from sklearn.linear_model import RidgeCV  
ridgeCV_model = RidgeCV(alphas=(0.1, 1.0, 10.0), scorers= 'neg_mean_absolute_error')  
ridgeCV_model.fit(X_train,y_train)  
test_predictions = ridgeCV_model.predict(X_test)</font>

Смотрим, какой же альфа лучший  
<font color="yellow">ridge*cv_model.alpha*</font>  
Дальше можно им поиграть в окрестности полученного значения

Далее,  
<font color="yellow">test_predictions = ridge_cv_model.predict(X_test)  
from sklearn.metrics import mean_absolute_error,mean_squared_error</font>  
Кстати, интересно сравнить ошибку на тестовом и тренировочном наборах  
<font color="yellow">MAE_test = mean_absolute_error(y_test,test_predictions)  
MAE_train = mean_absolute_error(y_train,train_predictions)

MSE_test = mean_squared_error(y_test,test_predictions)
MSE_train = mean_absolute_error(y_train,train_predictions)

RMSE = np.sqrt(MSE)</font>  
Ну и посмотрим на коэффициенты  
<font color="yellow">ridge*cv_model.coef*  
ridge_cv_model.best_score</font>  
Но более полезны метрики ошибок выше

По сути,это та же линейная регрессия, но указываем коэффициенты альфа,
метрику и кросс-валидацию, далее оцениваем модель.

---

## <u>Регрессия Лассо</u>

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html

from sklearn.linear_model import LassoCV
lasso_cv_model = LassoCV(eps=0.1,n_alphas=100,cv=5)

Что такое eps, cv и n*alphas? eps, cv и n_alphas это развернутый параметр альфа:
eps - это мин*альфа/макс_альфа (по умолчанию = 1е-3),
а n_alphas - их к-во ((по умолчанию = 100)). cv - разбиение выборки на 5 частей.

lasso_cv_model.fit(X_train,y_train)

Если у нас обучение не случается, то можно увеличить к-во итераций, или увеличить эпсилон

Далее оценки как написано выше для **Ridge Regression**
Но, посколько мы знаем, что эта регуляризация обнуляет некоторые коэффициенты, то на них следует взглянуть
lasso*cv_model.coef*

---

## <u>Elastic Net</u>

<font color="yellow">from sklearn.linear*model import ElasticNetCV  
elastic_model = ElasticNetCV(l1_ratio=[.1, .5, .7,.9, .95, .99, 1],tol=0.01)
sklearn.linear_model import ElasticNetCV  
elastic_model = ElasticNetCV(l1_ratio=[.1, .5, .7,.9, .95, .99, 1],eps=0.001, n_alphas=100, max_iter=1000000)  
elastic_model.fit(X_train,y_train)  
elastic_model.l1_ratio*
elastic*model.alfa*
elastic*model.coef*</font>  
Кстати, если elastic*model.l1_ratio* = 1
то значит это чистая модель ЛАССО

Далее все стандартно ...

### <font color="lightgrey">4.8. Внедрение (сохранение, развертывание) модели</font>

<font color="lightgreen">_Обучаем модель на полных даннных_</font>  
<font color="yellow">final*model = LinearRegression()  
 final_model.fit(X, y)  
 from joblib import dump, load  
 dump(final_model, 'name_model.joblib')<font color="lightgreen"> *- cохраняем модель в файл*</font>  
 loaded_model = load('name_model.joblib') <font color="lightgreen"> *- достаем модель из файла*</font>  
 loaded_model.coef* <font color="lightgreen"> _- смотрим, какой формат данных должен быть при подачи данных на нее. Например - array[[0.04, 0.2, 0.01]], значит это (ххх, 3), соответсвенно, форма подаваемых данных должна быть вида:_</font>  
 campaign = [[22, 34, 12]]  
 loaded_model.predict(campaign)</font>

---

# Приложения

## 1. <u>Полиномиальная регрессия</u>

Сначала, как обычно, разбиваем данные на признаки и на ЦП  
<font color="yellow">X = df.drop('sales',axis=1)  
y = df['sales']</font>

Создаем модель создания признаков  
<font color="yellow">from sklearn.preprocessing import PolynomialFeatures  
polynomial_converter = PolynomialFeatures(<font color="orange">degree=2,include_bias=False</font>)</font>

Параметры:

<font color="orange">interaction_only : bool, default=False</font>

If **interaction_only = True**

-   included: x[0], x[1], x[0] \_ x[1], etc.
-   excluded: x[0] ** 2, x[0] ** 2 \_ x[1], etc.

<font color="orange">include_bias : bool, default=True</font>  
If **include_bias =True**, then include a bias column, the feature in which all polynomial powers are zero (i.e. a column of ones - acts as an intercept term in a linear model). Т.е., в этом случае мы учитываем смещение, или, учитываем, что при нулевых значениях признаков целевая переменная уже чему то равна.

Далее мы вызываем метод fit, но он не выполняет действий ни по обучению, ни по созданию полиномиальных признаков,
а только выполняет анализ исходных признаков  
<font color="yellow">poly_features = polynomial_converter.fit(X)</font>  
а затем мы создаем эти признаки  
<font color="yellow">poly_features = polynomial_converter.transform(X)</font>

Хотя, все это можно сделать в одной операции  
<font color="yellow">poly_features = polynomial_converter.fit_transform(X)</font>

Можно посмотреть на к-во признаков  
Было:  
<font color="yellow">X.shape (X.iloc[0])</font>  
стало:  
<font color="yellow">poly_features.shape (poly_features[0])</font>

Разбиваем на тренировочную и тестовую выборки  
<font color="yellow">from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)</font>

Создаем модель линейной регрессии  
<font color="yellow">from sklearn.linear_model import LinearRegression  
model = LinearRegression(<font color="orange">fit_intercept=True</font>)</font>  
Здесь параметр:  
<font color="orange">fit_intercept : bool, default=True</font>  
Рассчитывать ли отсекаемый элемент для этой модели. Если установлено значение False, отсекаемый элемент не будет использоваться в расчетах (т. е. ожидается, что данные будут центрированы).

Обучаем  
<font color="yellow">model.fit(X_train,y_train)</font>  
Предсказываем  
<font color="yellow">test_predictions = model.predict(X_test)</font></font>  
Оцениваем

---

### Дилемма смещения-дисперсии (Bias-Variance Trade-off): недообучение или переобучение

---

Пример - поиск лучшей полиномиальной модели  
<font color="lightgreen">1. Создадим полиномиальные данные некоторой степени для данных X 2. Разобъём полиномиальные данные для обучающий и тестовый наборы данных 3. Выполним обучение модели на обучающем наборе данных 4. Посчитаем метрики на обучающем _и_ тестовом наборе данных 5. Нанесём эти данные на график, чтобы увидеть момент переобучения модели</font>

<font color="yellow">train_rmse_errors = []  
test_rmse_errors = []

for d in range(1,10):  
polynomial_converter = PolynomialFeatures(degree=d,include_bias=False)  
poly_features = polynomial_converter.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)  
model = LinearRegression(fit_intercept=True)  
model.fit(X_train,y_train)  
train_pred = model.predict(X_train)  
test_pred = model.predict(X_test)
train_RMSE = np.sqrt(mean_squared_error(y_train,train_pred))  
test_RMSE = np.sqrt(mean_squared_error(y_test,test_pred))  
 train_rmse_errors.append(train_RMSE)  
 test_rmse_errors.append(test_RMSE)</font>  
Строим график  
<font color="yellow">plt.plot(range(1,6),train_rmse_errors[:5],label='TRAIN')  
plt.xlabel("Polynomial Complexity")  
plt.ylabel("RMSE")  
plt.legend()</font>

Сохранение/внедрение модели

1. Выбираем финальные значения параметров на основе тестовых метрик
2. Выполняем повторное обучение на всех данных
3. Сохраняем объект Polynomial Converter
4. Сохраняем модель

<font color="yellow">final_poly_converter = PolynomialFeatures(degree=3,include_bias=False)  
final_model = LinearRegression()  
final_model.fit(final_poly_converter.fit_transform(X),y)  
from joblib import dump, load  
dump(final_model, 'sales_poly_model.joblib')  
dump(final_poly_converter,'poly_converter.joblib')

loaded_poly = load('poly_converter.joblib')  
loaded_model = load('sales_poly_model.joblib')  
campaign = [[149,22,12]]  
campaign_poly = loaded_poly.transform(campaign)  
final_model.predict(campaign_poly)</font>

---

---

#### 2. Визуализация предсказанных и точных значений (если признаков 2-3)

<font color="yellow">y_hat = final_model.predict(X)  
 fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(16,6))
axes[0].plot(df['1_feature'],df['TARGET'],'o')  
 axes[0].plot(df['1_feature'],y_hat,'o',color='red')  
 axes[0].set_ylabel("TARGET")  
 axes[0].set_title("1_feature Spend")  
 axes[1].plot(df['2_feature'],df['TARGET'],'o')  
 axes[1].plot(df['2_feature'],y_hat,'o',color='red')  
 axes[1].set_title("2_feature Spend")  
 axes[1].set_ylabel("TARGET")  
 axes[2].plot(df['3_feature'],df['TARGET'],'o')  
 axes[2].plot(df['3_feature'],y_hat,'o',color='red')  
 axes[2].set_title("3_feature Spend");  
 axes[2].set_ylabel("TARGET")  
 plt.tight_layout();</font>

Кстати!!
<font color = "yellow">ConfusionMatrixDisplay.from_estimator(model,X_test,y_test)</font>
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
