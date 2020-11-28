import pandas
import re
from scipy import stats

df = pandas.read_csv("titanic.csv")

#4. Какого возраста были пассажиры? Посчитайте среднее и медиану возраста пассажиров. Посчитайте среднее и медиану возраста
#пассажиров. В качестве ответа приведите два числа через пробел.
age_mean = df['Age'].mean()
age_mediana = df['Age'].median()
print(age_mean, age_mediana)