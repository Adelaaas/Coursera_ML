import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("D:\study\я_профессионал\курс\\titanic.csv")

df1 = df.drop(['PassengerId', 'Name', 'SibSp',
       'Parch', 'Ticket', 'Cabin', 'Embarked'], axis = 1)

df1['Sex'] = LabelEncoder().fit_transform(df1['Sex'])
df1 = df1.dropna()
print(df1.isna().sum())
y = df1['Survived'].values
df1.drop(['Survived'], axis=1, inplace=True)

y = np.array(y)
X = df1.iloc[:, [0,1,2,3]].values

clf = DecisionTreeClassifier()
clf.fit(X,y)

importances = clf.feature_importances_
print(importances)