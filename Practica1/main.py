#Imports
import pandas as pd


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