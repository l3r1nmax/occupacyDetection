import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import timeit
from matplotlib.finance import date2num
from sklearn import datasets
from sklearn import tree

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

def my_range(start, end, step):
    while start <= end:
        yield start
        start += step

heldout = [0.95, 0.90, 0.75, 0.50, 0.01]
rounds = 20
columnNames = ["id","Temperature","Humidity","Light","CO2","HumidityRadio","Occupancy"]
occ = pd.read_csv("/Users/lerin/Documents/Uni/TUWien/Machine Learning/Exercise 1/Occupancy/datatraining.csv", names = columnNames)
occTest = pd.read_csv("/Users/lerin/Documents/Uni/TUWien/Machine Learning/Exercise 1/Occupancy/datatest.csv", names = columnNames)
X, y = occ.drop('id',axis=1).drop('Occupancy',axis=1)[1:], occ['Occupancy'][1:]
X_test, y_test = occTest.drop('id',axis=1).drop('Occupancy',axis=1)[1:], occTest['Occupancy'][1:]
dep = [1,5,10,15,20,40,60,80,100,150,200]

for x in dep:

    classifiers = [("Random forest: " + str(x), RandomForestClassifier(n_estimators=x))]

    xx = range(1,5)
    yy = []
    zz = []

    for name, clf in classifiers:
        print("training %s" % name)
        rng = np.random.RandomState(42)
        yy_ = []
        start = timeit.default_timer()
        for r in range(rounds):
            clf.fit(X, y)
            y_pred = clf.predict(X_test)
            #prediction_dataframe = pd.DataFrame(data=y_pred, index=y_test.index, columns=['Malicious'])
            #test_dataframe = pd.DataFrame(data=y_test, index=y_test.index, columns=['Malicious'])
            #prediction_dataframe.to_csv("solution-" + str(i) + "-" + str(r) + ".csv")
            yy_.append(1 - np.mean(y_pred == y_test))
        stop = timeit.default_timer()
        yy.append(np.mean(yy_))
        zz.append(stop - start)
    print("Trees " + str(x))
    print("Error ratio " + str(yy))
    print("Runtimes " + str(zz))
plt.bar(xx, yy, label=name)

plt.legend(loc="upper right")
plt.xlabel("Algorithms")
plt.ylabel("Test Error Rate")
plt.show()
