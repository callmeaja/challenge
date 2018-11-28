import numpy as np
from nltk.corpus import stopwords

s = set(stopwords.words('english'))
GloveEmbeddings = {}
emb_dim = 50

# Map embeddings to each word from a pre trained model


def load_embeddings(embeddingfile):
    global GloveEmbeddings, emb_dim
    fe = open(embeddingfile, "r", encoding="utf-8", errors="ignore")
    for line in fe:
        tokens = line.strip().split()
        word = tokens[0]
        vec = [float(a) for a in tokens[1:]]
        GloveEmbeddings[word] = vec
    fe.close()

    return

# Get the array representation of each text passage / query


def concat_embeddings(text, query=False, pad_size=emb_dim, max_len=50):
    # import nltk
    # p = nltk.PorterStemmer()

    GloveEmbeddings['zerovec'] = [0]*pad_size
    if not query:
        filtered_text = list(filter(lambda w: not w in s, text.lower().split()))
        remaining = max_len - len(filtered_text)
        if remaining > 0:
            filtered_text += ['zerovec']*remaining
        else:
            filtered_text = filtered_text[:max_len]
    else:
        filtered_text = text.lower().split()
      

    processed = ["".join(list(filter(str.isalnum, text))) for text in filtered_text]
    # singularize = [p.stem(word) for word in processed]
    vector_array = []

    for i, word in enumerate(processed):
        if word in GloveEmbeddings:
            vector_array += GloveEmbeddings[word]
        else:
            vector_array += GloveEmbeddings['zerovec']

    return vector_array

