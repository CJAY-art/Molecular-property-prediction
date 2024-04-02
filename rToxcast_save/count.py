import pandas as pd
import math
import csv

data1 = pd.read_csv('test.csv')
data2 = pd.read_csv('test_scores.csv')

counts=[]
tfs=[]
for index,row in data2.iterrows():
    task_name=row['Task']
    count=data1[task_name].notna().sum()
    t=data1[task_name].dropna().sum()
    f=count-t
    counts.append(count)
    tfs.append(t/(f+1))
data2.insert(data2.shape[1], 'valid_count', counts)
data2.insert(data2.shape[1], 'T/F', tfs)
data2.to_csv('test_scores.csv')
