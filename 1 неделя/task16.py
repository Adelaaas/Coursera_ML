import pandas
import re
from scipy import stats

df = pandas.read_csv("titanic.csv")

# 6. Какое самое популярное женское имя на корабле? Извлеките из
# полного имени пассажира (колонка Name) его личное имя (First Name). 
rexp = re.findall(r'(?<=Mrs. ).*| (?<=Miss. ).*', df['Name'].to_string())
str1 = ' '
str1 = str1.join(rexp)
str1 = re.sub(r'["()"...]','', str1).split()
name = stats.mode(str1)
print(name.mode[0])