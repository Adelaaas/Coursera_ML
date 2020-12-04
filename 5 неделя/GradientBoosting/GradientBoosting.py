from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import operator, os
from sklearn.ensemble import RandomForestClassifier

def sigma(y):
    return 1 / (1 + np.exp(-y))

def plot(train, test, name_postfix):
    plt.figure()
    plt.plot(train, 'r', linewidth=3)
    plt.plot(test, 'b', linewidth=3)
    plt.legend(['train', 'test'])
    plt.savefig('D:\study\я_профессионал\GradientBoosting' + str(name_postfix) + '.png')

df = pd.read_csv("D:\study\я_профессионал\GradientBoosting\gbm-data.csv")
print(df)

X = df.iloc[:,1:].values
y = df.iloc[:,0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

print('X_train: ', X_train.shape)
print('X_test: ', X_test.shape)
print('y_train: ', y_train.shape)
print('y_test: ', y_test.shape)

# Обучите GradientBoostingClassifier с параметрами n_estimators=250, verbose=True, random_state=241 и 
# для каждого значения learning_rate из списка [1, 0.5, 0.3, 0.2, 0.1] проделайте следующее:

learning_rates = np.array([1, 0.5, 0.3, 0.2, 0.1])

for learning_rate in learning_rates:
    # создаем градиентный бустинг и обучаем его для каждого значения learning_rate
    clf = GradientBoostingClassifier (learning_rate=learning_rate, n_estimators=250, verbose=True, random_state=241)
    clf.fit(X_train, y_train)

    # что тут происходит????

    # Compute decision - function of X for each iteration.
    y_pred_train = [sigma(score) for score in clf.staged_decision_function(X_train)]
    y_pred_test = [sigma(score) for score in clf.staged_decision_function(X_test)]

    # count log_loss
    train_loss = np.array([log_loss(y_train, y_pred) for y_pred in y_pred_train])
    test_loss = np.array([log_loss(y_test, y_pred) for y_pred in y_pred_test])

    plot(train_loss, test_loss, learning_rate)
    min_loss_index, min_loss = min(enumerate(test_loss), key=operator.itemgetter(1))

    # Приведите минимальное значение log-loss на тестовой выборке и номер итерации,
    # на котором оно достигается, при learning_rate = 0.2.
    if learning_rate == 0.2:
        print('Answer:', round(min_loss,2), min_loss_index)
        
        # 3. Как можно охарактеризовать график качества на тестовой выборке,
        # начиная с некоторой итерации: переобучение (overfitting) или
        # недообучение (underfitting)?
        #
        # По графику можно видеть, что хотя на обучающей выборке качество возрастает,
        # на тестовой выборке после примерно 50 итерации качество начинает убывать.
        fitting = 'overfitting' if test_loss[int(3.*test_loss.size/4.) :].mean() > test_loss.mean() else 'underfitting'
        print('Fitting', fitting)


# На этих же данных обучите RandomForestClassifier с количеством деревьев, равным количеству итераций,
# на котором достигается наилучшее качество у градиентного бустинга из предыдущего пункта, c random_state=241 и остальными параметрами по умолчанию. Какое значение log-loss на тесте получается у этого случайного леса? (Не забывайте, что предсказания нужно получать с помощью функции predict_proba.
# В данном случае брать сигмоиду от оценки вероятности класса не нужно)

forest = RandomForestClassifier(n_estimators=min_loss_index, random_state=241)
forest.fit(X_train, y_train)

y_forest = forest.predict_proba(X_test)
forest_loss = log_loss(y_test, y_forest)

print("forest logloss:", round(forest_loss,2))
