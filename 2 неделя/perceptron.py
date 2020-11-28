import numpy as np
from sklearn.linear_model import Perceptron
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df_train = pd.read_csv("C:\\Users\\adela\\Desktop\\я_профессионал\\Coursera_ML\\2 неделя\\perceptron-train.csv", header=None)
df_test = pd.read_csv("C:\\Users\\adela\\Desktop\\я_профессионал\\Coursera_ML\\2 неделя\\perceptron-test.csv", header=None)

X_train = df_train.iloc[:, [1,2]].values
y_train = df_train.iloc[:, 0].values

X_test = df_test.iloc[:, [1,2]].values
y_test = df_test.iloc[:, 0].values

clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
a1 = accuracy_score(y_test,predictions)
print("Accuracy без нормализации", a1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

clf = Perceptron(random_state=241)
clf.fit(X_train_scaled, y_train.ravel())
predictions = clf.predict(X_test_scaled)
a2 = accuracy_score(y_test.ravel(),predictions)
print("Accuracy с нормализации", a2)

print(round((a2-a1),3))