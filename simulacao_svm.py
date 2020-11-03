

import pandas as pd 
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import string
import numpy as np

#conda install -c anaconda biopython
#from Bio.Seq import Seq
#from Bio.Seq import UnknownSeq


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




def compute_freq(seq):
    features=[] 
    diamino=[]
   
    
    for n1 in alphabet:
            for n2 in alphabet:
                diamino.append(n1+n2)        
        
    l=len(alphabet)     
    for n in alphabet:
        freq = seq.count(n)  / l        
        features.append(freq)

    for d in diamino: 

        # d is one of the dinucleotides
        fx=seq.count(d[0])/l # d position [0] is the first nucleotide
        fy=seq.count(d[1])/l # d position [1] is the second nucleotide

        # start a counter with 0
        fxy=0 
        # we will travel the entire length of the sequence to the penultimate position
        for i in range(len(seq)-1): #  range starts in zero until l-1 that is the penultimate position
            if seq[i]+seq[i+1]==d: 
                fxy+=1 # add one to the counter
        fxy=fxy/l # Do not forget to divide the count by the length
        
        if(fx*fy == 0):
            features.append(0)
        else:
            features.append(fxy/(fx*fy))
            
    return features

def run_experiment(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)
    
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 1000],  
                  'gamma': [1000,100, 10, 1, 0.1, 0.01, 0.001], 
                  'kernel': ['rbf', 'sigmoid' ,'linear']}
    
    grid_search = GridSearchCV(SVC(), param_grid, scoring="accuracy", verbose = 3, n_jobs=-1, cv = 5)
    grid_search.fit(X_train, y_train)  
    classifier = grid_search.best_estimator_
    
    
    y_pred = classifier.predict(X_test)
    plot_confusion_matrix(classifier, X_test, y_test, normalize = 'true')
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))


dataset = pd.read_csv("1625Data.txt", sep=",")

dataset["CLASSE"]  = dataset["CLASSE"] == 1
dataset["CLASSE"]  = dataset["CLASSE"].astype(int)

y = dataset["CLASSE"].values


#Experiment 1 - Following the author's methodology
patterns = dataset["PADRAO"].values

X = []

for p in patterns:  
    X.append(convert_to_onehot(p))

X = np.vstack(X)

run_experiment(X, y)

#Experiment 2 - #Experiment 1 - Following the author's methodology
X = []

for p in patterns:  
    X.append(compute_freq(p))

X = np.vstack(X)
X = StandardScaler().fit_transform(X)
run_experiment(X,y)