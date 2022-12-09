import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
#https://scikit-learn.org/stable/modules/model_evaluation.html
import matplotlib.pyplot as plt

import numpy as np
from numpy import mean

from sklearn.svm import SVC
from sklearn.svm import LinearSVC 


#Auxiliar function that reformats df to one that is more fitting to Classification Problems
def reformat(dataFrame):
    dataFrame['y']=dataFrame.columns[0]
    dataFrame.rename(columns={dataFrame.columns[0]: 'word'}, inplace=True)
    return dataFrame;
#RawData, it has been a little bit formatted before reading the csv, to facilitate the process.
raw=pd.read_csv("data/data.csv")

#Separate df
catala, angles= raw.filter(['catala'], axis=1), raw.filter(['angles'], axis=1)
#Reformating
catala=reformat(catala)
angles=reformat(angles)
#Merging
wordsDF=pd.concat([catala,angles], axis=0)
#Shuffle the rows
wordsDF = wordsDF.sample(frac=1).reset_index(drop=True)


#ratio de consonantes y vocales
def ratio (word):
    vocals=0
    for c in word:
        if isVocal(c):
            vocals+=1
    return vocals/len(word) 

#isVocal?
def isVocal(c):
    if(c=='a' or c=='e' or c=='i' or c=='o' or c=='u'):
        return True
    return 
#gotAccent?
def gotAccent(word):
    #List containing all possible accentuated chars from Catalan
    accentuatedChars=[ord('à'),ord('è'),ord('é'),ord('í'),ord('ò'),ord('ó'),ord('ú')]
    for c in word:
        if ord(c) in accentuatedChars:
            return 1
    return 0
def doubleVocal(word):
    ocurrences=["aa","ee","ii","oo","uu"]
    for oc in ocurrences:
        if word.find(oc)!=-1:
            return 1
    return 0

def enCC(word):
    ocurrences=["sch","spl","shr","squ","thr","spr","scr","sph","th","tw","sw","sk","sm"]
    for oc in ocurrences:
        if word.find(oc)!=-1:
            return 1
    return 0
'''
def doubleVocal(word):
    possibleDuo=False
    vocal = ' '
    for c in word:
        if possibleDuo==True:
            #Possible Duo Found?
            if isVocal(c):
                if c == vocal:
                    #Duo Found
                    return 1
                else:
                    #New Possible Duo
                    vocal = c
            #Not Found 
            else:
                possibleDuo=False
        else:
            if isVocal(c):
                #Possible Duo in the future
                vocal=c
                possibleDuo=True
    return 0
'''


wordsDF['ratio']=wordsDF['word'].apply(ratio)
wordsDF['cantidadLetras']=wordsDF['word'].apply(len)
wordsDF['gotAccent']=wordsDF['word'].apply(gotAccent)
wordsDF['doubleVocal']=wordsDF['word'].apply(doubleVocal)
wordsDF['enCC']=wordsDF['word'].apply(enCC)
wordsDF=wordsDF[['word','ratio','cantidadLetras','gotAccent','doubleVocal','enCC','y']]
wordsDF.to_csv('data/definitiveData.csv', index=False)
print(wordsDF)

#First we must separate dataframe into X and y format
X=wordsDF.iloc[:,1:3]
y=wordsDF.iloc[:,6]
print(X)
#Then we separate the data frame in training and test (will be used in chosen model)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)
#Modelo
SVM_Lineal = LinearSVC(loss='squared_hinge', dual=False, C=50)

def evaluate_model(cv):
    #https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
    score = cross_val_score(SVM_Lineal,X,y,scoring='accuracy',cv=cv,n_jobs=-1)
    return mean(score),score.min(),score.max()


#Range to be tested
folds = range (5,11)
for k in folds:
    cv=StratifiedKFold(n_splits=k, shuffle=True)
    k_mean, k_min, k_max = evaluate_model(cv)
    print(f'-> folds={k}, accuracy = {round(k_mean,4)}, ({round(k_min,4)}, {round(k_max,4)})')