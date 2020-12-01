import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA

# Загрузите данные close_prices.csv. В этом файле приведены цены акций 30 компаний на закрытии торгов за каждый день периода.
djia_index = pd.read_csv("D:\study\я_профессионал\PCA\djia_index.csv")
close_prices = pd.read_csv("D:\study\я_профессионал\PCA\close_prices.csv")
print(close_prices)

X = close_prices.iloc[:, 1:]
# На загруженных данных обучите преобразование PCA с числом компоненты равным 10. 
# Скольких компонент хватит, чтобы объяснить 90% дисперсии?
pca = PCA(n_components=10)
transformed_data = pca.fit_transform(X)
ratio = pca.explained_variance_ratio_
ratio_90 = 0
print(ratio)
i = 0
while ratio_90 < 0.9:
    ratio_90 += ratio[i]
    i += 1
print(ratio_90, i)

# Примените построенное преобразование к исходным данным и возьмите значения первой компоненты.
first_component = transformed_data[:, 0]

# Загрузите информацию об индексе Доу-Джонса из файла djia_index.csv.
# Чему равна корреляция Пирсона между первой компонентой и индексом Доу-Джонса?
corrcoef = np.corrcoef(first_component, djia_index.iloc[:, 1])[0, 1]
print(round(corrcoef,2))

# Какая компания имеет наибольший вес в первой компоненте? Укажите ее название с большой буквы.
company = X.keys()[pca.components_[0].argmax()]
print(company)