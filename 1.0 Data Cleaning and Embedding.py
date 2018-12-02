import pandas as pd
import os
os.chdir('E:/Competitions/Microsoft AI challenge')

execfile('Codes/0.0 Init.py')
import numpy as np


# Reading the data
data = pd.read_csv('Data/data.tsv', delimiter='\t', header=None)
data.columns = ['query_id', 'query', 'passage_text', 'label', 'passage_id']

data = data[:100000]
# Loading embeddings
load_embeddings('glove.6B.50d.txt')

data['query'] = data['query'].apply(concat_embeddings, query=True)
data['passage_text'] = data['passage_text'].apply(concat_embeddings)

