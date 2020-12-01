import pandas as pd 
from sklearn.svm import SVC

df = pd.read_csv("D:\study\я_профессионал\курс\svm-data.csv", header = None)

y = df.iloc[:, 0].values
X = df.iloc[:, [1,2]].values

clf = SVC(kernel='linear', C = 100000, random_state=241)
clf.fit(X, y)
n_sv = clf.support_

print(n_sv)