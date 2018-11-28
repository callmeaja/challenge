import pandas as pd
execfile('Codes/0.0 Init.py')
import numpy as np


# Reading the data
data = pd.read_csv('Data/data.tsv', delimiter='\t', header=None)
data.columns = ['query_id', 'query', 'passage_text', 'label', 'passage_id']

# Loading embeddings
load_embeddings('glove.6B.50d.txt')