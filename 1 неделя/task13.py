import pandas
import re
from scipy import stats

df = pandas.read_csv("titanic.csv")

# 3. Какую долю пассажиры первого класса составляли среди всех пассажиров? Ответ приведите в процентах
# (число в интервале от 0 до 100, знак процента не нужен), округлив до двух знаков.
all = df['Survived'].count()
pass_in_1_class = df[df.Pclass == 1].shape[0]
print(round((pass_in_1_class/all)*100,2))