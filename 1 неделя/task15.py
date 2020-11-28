import pandas
import re
from scipy import stats

df = pandas.read_csv("titanic.csv")

# 5. Коррелируют ли число братьев/сестер с числом родителей/детей?
# Посчитайте корреляцию Пирсона между признаками SibSp и Parch
print(df['SibSp'].corr(df['Parch']))