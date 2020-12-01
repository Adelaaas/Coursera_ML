import pandas as pd 
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Загрузите объекты из новостного датасета 20 newsgroups, относящиеся к категориям "космос" и "атеизм" (инструкция приведена выше). 
# Обратите внимание, что загрузка данных может занять несколько минут
newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])
X = newsgroups.data
y = newsgroups.target

# Вычислите TF-IDF-признаки для всех текстов. Обратите внимание, что в этом задании мы предлагаем вам вычислить TF-IDF по всем данным. 
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)
feature_map = vectorizer.get_feature_names()

# Подберите минимальный лучший параметр C из множества [10^-5, 10^-4, ... 10^4, 10^5] для SVM с линейным ядром (kernel='linear')
# при помощи кросс-валидации по 5 блокам. Укажите параметр random_state=241 и для SVM, и для KFold.
# В качестве меры качества используйте долю верных ответов (accuracy).
grid = {'C': np.power(10.0, np.arange(-6,6))}
kf = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=kf)
gs.fit(X,y)

C = gs.best_params_
# score = 0
# C = 0
# for attempt in gs.cv_results_:
#     if attempt.mean_validation_score > score:
#         score = attempt.mean_validation_score
#         C = attempt.parameters['C']

# Обучите SVM по всей выборке с оптимальным параметром C, найденным на предыдущем шаге.
clf = gs.best_estimator_
clf.fit(X, y)

# Найдите 10 слов с наибольшим абсолютным значением веса (веса хранятся в поле coef_ у svm.SVC). Они являются ответом на это задание.
# Укажите эти слова через запятую или пробел, в нижнем регистре, в лексикографическом порядке.
weights = np.absolute(clf.coef_.toarray())

max_weights = sorted(zip(weights[0], feature_map))[-10:]
max_weights.sort(key=lambda x: x[1])
print(max_weights)