import pandas
import re
from scipy import stats

df = pandas.read_csv("C:\Users\Adele\Desktop\\titanic.csv")

# 1. Какое количество мужчин и женщин ехало на корабле? В качестве
# ответа приведите два числа через пробел.
male = df[df['Sex'] == 'male']
female = df[df['Sex'] == 'female']
print(len(male), len(female))