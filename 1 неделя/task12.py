import pandas
import re
from scipy import stats

df = pandas.read_csv("titanic.csv")

# 2. Какой части пассажиров удалось выжить? Посчитайте долю выживших пассажиров.
# Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен), округлив до двух знаков
all = df['Survived'].count()
survived = df[df.Survived > 0].shape[0]
print(round((survived/all)*100, 2))