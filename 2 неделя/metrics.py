from sklearn.datasets import load_boston
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score

df = load_boston()

y = df.target
features = df.data
X = preprocessing.scale(features)

# p = np.linspace(1,10,num=200)
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# p_score = []
# for p_i in p:
#     knn = KNeighborsClassifier(n_neighbors=5, weights='distance', p=p_i, metric='minkowski')
#     cvs = cross_val_score(estimator=knn, X=X, y=y, cv=kf, scoring='neg_mean_squared_error',error_score ='raise')
#     mean = cvs.mean()
#     p_score.append(mean)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
cvs = cross_val_score(estimator=knn, X=features, y=y, cv=kf)
print(cvs.mean())
# best_p = max(p_score)
# print(p_score)
# print("максимальное число",best_p)
# print("длинна списка", len(p_score))
# for i in range(len(p_score)):
#     if p_score[i] == best_p:
#         print(i, p_score[i])


# file_answer = open("metric.txt", "w")
# file_answer.write(repr(round(best_p, 1)))
# file_answer.close()