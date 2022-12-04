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
wordsDF.to_csv('data/definitiveData.csv', index=False)
