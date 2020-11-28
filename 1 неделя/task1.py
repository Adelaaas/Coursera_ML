import pandas
import re
from scipy import stats

df = pandas.read_csv("D:/study/я_профессионал/курс/titanic.csv")

# 1. Какое количество мужчин и женщин ехало на корабле? В качестве
# ответа приведите два числа через пробел.
male = df[df['Sex'] == 'male']
female = df[df['Sex'] == 'female']
print(len(male), len(female))

# 2. Какой части пассажиров удалось выжить? Посчитайте долю выживших пассажиров.
# Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен), округлив до двух знаков
all = df['Survived'].count()
survived = df[df.Survived > 0].shape[0]
print(round((survived/all)*100, 2))

# 3. Какую долю пассажиры первого класса составляли среди всех пассажиров? Ответ приведите в процентах
# (число в интервале от 0 до 100, знак процента не нужен), округлив до двух знаков.
pass_in_1_class = df[df.Pclass == 1].shape[0]
print(round((pass_in_1_class/all)*100,2))

#4. Какого возраста были пассажиры? Посчитайте среднее и медиану возраста пассажиров. Посчитайте среднее и медиану возраста
#пассажиров. В качестве ответа приведите два числа через пробел.
age_mean = df['Age'].mean()
age_mediana = df['Age'].median()
print(age_mean, age_mediana)

# 5. Коррелируют ли число братьев/сестер с числом родителей/детей?
# Посчитайте корреляцию Пирсона между признаками SibSp и Parch
print(df['SibSp'].corr(df['Parch']))

# 6. Какое самое популярное женское имя на корабле? Извлеките из
# полного имени пассажира (колонка Name) его личное имя (First Name). 
rexp = re.findall(r'(?<=Mrs. ).*| (?<=Miss. ).*', df['Name'].to_string())
str1 = ' '
str1 = str1.join(rexp)
str1 = re.sub(r'["()"...]','', str1).split()
name = stats.mode(str1)
print(name.mode[0])