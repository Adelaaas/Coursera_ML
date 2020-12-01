import pandas as pd 
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import coo_matrix, hstack

df_train = pd.read_csv("D:\study\я_профессионал\курс\linear regression\salary-train.csv")
df_test = pd.read_csv("D:\study\я_профессионал\курс\linear regression\salary-test-mini.csv")
print(df_train)

# Приведите тексты к нижнему регистру (text.lower())
df_train['FullDescription'] = df_train['FullDescription'].apply(lambda x: x.lower())
print(df_train)
print(df_train.isna().sum())

# Замените пропуски в столбцах LocationNormalized и ContractTime на специальную строку 'nan'. Код для этого был приведен выше.
df_train['LocationNormalized'].fillna('nan', inplace=True)
df_train['ContractTime'].fillna('nan', inplace=True)
print(df_train.isna().sum())

# Замените все, кроме букв и цифр, на пробелы — это облегчит дальнейшее разделение текста на слова. 
# Для такой замены в строке text подходит следующий вызов: re.sub('[^a-zA-Z0-9]', ' ', text). 
# Также можно воспользоваться методом replace у DataFrame, чтобы сразу преобразовать все тексты:
df_train['FullDescription'] = df_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
df_test['FullDescription'] = df_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
print(df_train)

# Примените TfidfVectorizer для преобразования текстов в векторы признаков. 
# Оставьте только те слова, которые встречаются хотя бы в 5 объектах (параметр min_df у TfidfVectorizer).
vectorizer = TfidfVectorizer(min_df=5)
tf_idf = vectorizer.fit_transform(df_train['FullDescription'])
tf_idf_test = vectorizer.transform(df_test['FullDescription'])

# Примените DictVectorizer для получения one-hot-кодирования признаков LocationNormalized и ContractTime
enc = DictVectorizer()
X_train_categ = enc.fit_transform(df_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(df_test[['LocationNormalized', 'ContractTime']].to_dict('records'))


# Объедините все полученные признаки в одну матрицу "объекты-признаки".
# Обратите внимание, что матрицы для текстов и категориальных признаков являются разреженными.
# Для объединения их столбцов нужно воспользоваться функцией scipy.sparse.hstack.
X = hstack([tf_idf, X_train_categ])
X_test = hstack([tf_idf_test, X_test_categ])
y = df_train['SalaryNormalized']

# Обучите гребневую регрессию с параметрами alpha=1 и random_state=241. Целевая переменная записана в столбце SalaryNormalized
model = Ridge(alpha=1, random_state=241)
model.fit(X,y)

# 4. Постройте прогнозы для двух примеров из файла salary-test-mini.csv.
# Значения полученных прогнозов являются ответом на задание. Укажите их через пробел.
res = model.predict(X_test)
print(round(res[0],2), round(res[1],2))