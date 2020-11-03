

import pandas as pd 
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import string
import numpy as np



alphabet="ARNDCQEGHILKMFPSTWYV"
def convert_to_onehot(data):    
    char_to_int = dict((c,i) for i,c in enumerate(alphabet))
    encoded_data = []
    encoded_data.append([char_to_int[char] for char in data])    
    one_hot = []
    for value in encoded_data:        
        letter = [0 for _ in range(len(alphabet))]
        for v in value:
            letter[v] = 1
        one_hot.append(letter)
    return one_hot

dataset = pd.read_csv("1625Data.txt", sep=",")

dataset["CLASSE"]  = dataset["CLASSE"] == 1
dataset["CLASSE"]  = dataset["CLASSE"].astype(int)

y = dataset["CLASSE"].values

patterns = dataset["PADRAO"].values

X = []

for p in patterns:  
    X.append(convert_to_onehot(p))

X = np.vstack(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],  
              'gamma': [100, 10, 1, 0.1, 0.01, 0.001], 
              'kernel': ['rbf', 'sigmoid' ,'linear']}

grid_search = GridSearchCV(SVC(), param_grid, scoring="accuracy", verbose = 3, n_jobs=-1, cv = 5)
grid_result = grid_search.fit(X_train, y_train)  
classifier = grid_search.best_estimator_


y_pred = classifier.predict(X_test)
plot_confusion_matrix(classifier, X_test, y_test, normalize = 'true')
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
