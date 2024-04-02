import pandas as pd
import math
import csv

data1 = pd.read_csv('test.csv')
data2 = pd.read_csv('interpret.csv')

right=[]
for index,row in data2.iterrows():
    data2['smiles'][index]=eval(row['smiles'])[0]
    right.append(not ((data1['Label'][index])^(1 if row['Label']>0.5 else 0)))
data2.insert(data2.shape[1], 'right', right)
data2.to_csv('interpret1.csv')
