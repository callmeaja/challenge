import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
os.chdir('E:/Competitions/Microsoft AI challenge')

execfile('Codes/init.py')

train_test_split = 0.7
# Reading the data
data = pd.read_csv('Data/data.tsv', delimiter='\t', header=None)
data.columns = ['query_id', 'query', 'passage_text', 'label', 'passage_id']

data = data[:100000]
# Loading embeddings
load_embeddings('glove.6B.50d.txt')

data['query'] = data['query'].apply(concat_embeddings, query=True)
data['passage_text'] = data['passage_text'].apply(concat_embeddings)

ohe = OneHotEncoder(sparse=False)

x = data[['query', 'passage_text']]
y = ohe.fit_transform(np.array(data['label']).reshape(-1, 1))
del data

x_train = x.iloc[:int(train_test_split*len(x)), :]
x_test = x.iloc[int(train_test_split*len(x)):, :]
y_train = y[:int(train_test_split*len(x))]
y_test = y[int(train_test_split*len(x)):]

del x, y
# x_query = data['query'].apply(concat_embeddings, query=True)
# x_passage = data['passage_text'].apply(concat_embeddings)
# y_train = data['label']