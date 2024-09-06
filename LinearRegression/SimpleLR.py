import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbs

df = pd.read_csv('data/Advertising.csv')
# Combine the features into one - the sum of advertising across all sources
df['total_spend'] = df['TV'] + df['radio'] + df['newspaper']
# sbs.scatterplot(data = df, x = 'total_spend', y = 'sales')
# В seaborn есть метод построения длинейной регрессии
# sbs.regplot(data = df, x = 'total_spend', y = 'sales')

X = df['total_spend']
y = df['sales']
#  Строим ЛР y: = mx + b - B1*x + B0
B = np.polyfit(X, y, deg = 1)
# B: [B1, B0]
# print(B) # [0.04868788 4.24302822]
# ____________________________________________
potrntial_spend = np.linspace(0, 500, 100)
predict_sales = B[0]*potrntial_spend + B[1]
sbs.scatterplot(data = df, x = 'total_spend', y = 'sales')
plt.plot(potrntial_spend, predict_sales, color= 'red')
# print(predict_sales) 
# ____________________________________________
# Проба полиномиальной регрессии
C = np.polyfit(X, y, deg = 3)
# С: С3*X**3 + C2*X**2 + C1*x + C0
print(C) # [ 3.07615033e-07 -1.89392449e-04  8.20886302e-02  2.70495053e+00]
potetial_spend_polinom = np.linspace(0, 500, 100)
predict_sales_polinom = C[0]*potetial_spend_polinom**3 + \
                        C[1]*potetial_spend_polinom**2 + \
                        C[2]*potetial_spend_polinom +    \
                        C[3]
sbs.scatterplot(data = df, x = 'total_spend', y = 'sales')
plt.plot(potetial_spend_polinom, predict_sales_polinom, color= 'blue')
# ____________________________________________
plt.show()
