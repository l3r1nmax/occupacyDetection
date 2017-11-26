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

heldout = [0.95, 0.90, 0.75, 0.50, 0.01]
rounds = 20
columnNames = ["id","Temperature","Humidity","Light","CO2","HumidityRadio","Occupancy"]
occ = pd.read_csv("/Users/lerin/Documents/Uni/TUWien/Machine Learning/Exercise 1/Occupancy/datatraining.csv", names = columnNames)
X, y = occ.drop('id',axis=1).drop('Occupancy',axis=1)[1:], occ['Occupancy'][1:]

classifiers = [
    ("kNN", KNeighborsClassifier(n_neighbors = 3)),
    ("Random Forest", RandomForestClassifier(max_depth=2, random_state=0)),
    ("Perceptron", Perceptron()),
    ("Decision Tree", tree.DecisionTreeClassifier())
]

xx = 1. - np.array(heldout)

for name, clf in classifiers:
    print("training %s" % name)
    rng = np.random.RandomState(42)
    yy = []
    start = timeit.default_timer()
    for i in heldout:
        yy_ = []
        for r in range(rounds):
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=i, random_state=rng)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            #prediction_dataframe = pd.DataFrame(data=y_pred, index=y_test.index, columns=['Malicious'])
            #test_dataframe = pd.DataFrame(data=y_test, index=y_test.index, columns=['Malicious'])
            #prediction_dataframe.to_csv("solution-" + str(i) + "-" + str(r) + ".csv")
            yy_.append(1 - np.mean(y_pred == y_test))
        yy.append(np.mean(yy_))
        stop = timeit.default_timer()
    plt.plot(xx, yy, label=name + " " + str(stop - start) + "s")

plt.legend(loc="upper right")
plt.xlabel("Proportion train")
plt.ylabel("Test Error Rate")
plt.show()
