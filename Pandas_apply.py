# import numpy as np
# import pandas as pd

# df = pd.read_csv('./data/tips.csv')
# print(df)
# объявлении ф-ции, ктр возвращает последние 4 символа из строки
# с конвертацией в целые значения (можно не делать, если работаем 
# со строками)
# def last_four_nambers(num):
#     return int(str(num)[-4:])

# a = last_four_nambers(1234567890)
# Ф-ция вызывается методом apply(<name function>)
# print(df['CC Number'].apply(last_four_nambers))
# ______________________________________
# Rating restaurant
# $ - chipe, $$ - midle, $$$ - expensive
# def rating(price):
#     if price < 10:
#         return "$"
#     elif price >=10 and price <=30:
#         return "$$"
#     else:
#         return "$$$"

# df['rating'] = df['total_bill'].apply(rating)
# print(df[df['rating'] == "$$$"])    
# ______________ multiply colums _____________________________________
import timeit
setup = '''
import numpy as np
import pandas as pd
df = pd.read_csv('./data/tips.csv')
def qualityTip(total_bill, tip):
    if tip/total_bill >0.25:
        return "Handsome tip"
    else: return "Usual tip"
    '''
# option 1
stmt_one = '''
df['qualityTip'] = df[['total_bill', 'tip']].apply(lambda df: qualityTip(df['total_bill'], df['tip']), axis=1)
'''
# option 2
stmt_two = '''
df['qualityTip'] = np.vectorize(qualityTip)(df['total_bill'], df ['tip'])
'''
# print(df[df['qualityTip'] == 'Handsome tip'])
print(timeit.timeit(setup=setup, stmt = stmt_one, number = 1000))
print(timeit.timeit(setup=setup, stmt = stmt_two, number = 1000))
