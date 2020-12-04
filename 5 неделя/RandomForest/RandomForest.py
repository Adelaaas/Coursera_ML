import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score

# Загрузите данные из файла abalone.csv. Это датасет, в котором требуется предсказать возраст ракушки (число колец) по физическим измерениям.
df = pd.read_csv("D:\\study\\я_профессионал\\RandomForest\\abalone.csv")
print(df)

# Преобразуйте признак Sex в числовой: значение F должно перейти в -1, I — в 0, M — в 1.
# Если вы используете Pandas, то подойдет следующий код:
df['Sex'] = df['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
print(df)

# Разделите содержимое файлов на признаки и целевую переменную. В последнем столбце записана целевая переменная, в остальных — признаки.
X = df.iloc[:, [0,1,2,3,4,5,6,7]].values
y = df['Rings'].values

# Обучите случайный лес (sklearn.ensemble.RandomForestRegressor) с различным числом деревьев:
# от 1 до 50 (не забудьте выставить "random_state=1" в конструкторе).
# Для каждого из вариантов оцените качество работы полученного леса на кросс-валидации по 5 блокам.
# Используйте параметры "random_state=1" и "shuffle=True" при создании генератора кросс-валидации sklearn.cross_validation.KFold.
# В качестве меры качества воспользуйтесь коэффициентом детерминации (sklearn.metrics.r2_score).
kf = KFold(n_splits=5, shuffle=True, random_state=1)

for n in range(1,51):
    rf = RandomForestRegressor(n_estimators=n, random_state=1)
    score_kfold = np.array([])
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        rf.fit(X_train,y_train)
        y_pred = rf.predict(X_test)
        score_kfold = np.append(score_kfold, r2_score(y_test, y_pred))
    
    # score.append(sum(score_kfold)/len(score_kfold))
    current_score = score_kfold.mean()
    print('n_estimators: ', n, ' Score: ', current_score)
