import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn import preprocessing

data = np.loadtxt(r'D:\\study\\я_профессионал\\курс\\wine.data', delimiter=",")
X = data[:,1:14]
y = data[:,0]
print(X.shape)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

score = []
for i in range(1,51):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X,y)
    cvs = cross_val_score(estimator=knn, X=X, y=y, cv=kf, scoring='accuracy')
    mean = cvs.mean()
    score.append(mean)

opt_num_of_neib = max(score)
print(score)
print("максимальное число",opt_num_of_neib)
print("длинна списка", len(score))
for i in range(len(score)):
    if score[i] == opt_num_of_neib:
        print(i, score[i])

print("________________________________________")
X_scale = preprocessing.scale(X)
score = []
for i in range(1,51):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X,y)
    cvs = cross_val_score(estimator=knn, X=X_scale, y=y, cv=kf, scoring='accuracy')
    mean = cvs.mean()
    score.append(mean)

opt_num_of_neib = max(score)
print(score)
print("максимальное число",opt_num_of_neib)
print("длинна списка", len(score))
for i in range(len(score)):
    if score[i] == opt_num_of_neib:
        print(i, score[i])