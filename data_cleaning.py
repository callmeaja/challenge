import pandas as pd
import os
os.chdir('E:/Competitions/Microsoft AI challenge')

execfile('Codes/init.py')
import numpy as np

train_test_split = 0.7
# Reading the data
data = pd.read_csv('Data/data.tsv', delimiter='\t', header=None)
data.columns = ['query_id', 'query', 'passage_text', 'label', 'passage_id']

data = data[:100000]
# Loading embeddings
load_embeddings('glove.6B.50d.txt')

data['query'] = data['query'].apply(concat_embeddings, query=True)
data['passage_text'] = data['passage_text'].apply(concat_embeddings)

x = data[['query', 'passage_text']]
y = data['label']
del data

x_train = x.iloc[:int(0.6*len(x)), :]
x_test = x.iloc[int(0.6*len(x)):, :]
y_train = y[:int(0.6*len(x))]
y_test = y[int(0.6*len(x)):]

del x, y
# x_query = data['query'].apply(concat_embeddings, query=True)
# x_passage = data['passage_text'].apply(concat_embeddings)
# y_train = data['label']