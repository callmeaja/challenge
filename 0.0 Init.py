import numpy as np
from nltk.corpus import stopwords

s = set(stopwords.words('english'))
GloveEmbeddings = {}
emb_dim = 50
GloveEmbeddings['zerovec'] = [0]*emb_dim
# Map embeddings to each word from a pre trained model


def load_embeddings(embeddingfile):
    global GloveEmbeddings, emb_dim
    fe = open(embeddingfile, "r", encoding="utf-8", errors="ignore")
    for line in fe:
        tokens = line.strip().split()
        word = tokens[0]
        vec = tokens[1:]
        GloveEmbeddings[word] = vec
    fe.close()

    return

# Get the array representation of each text passage / query


def concat_embeddings(text, query=False, max_len_p=50, max_len_q=10):
    # import nltk
    # p = nltk.PorterStemmer()
    if not query:
        filtered_text = list(filter(lambda w: not w in s, text.lower().split()))
        remaining = max_len - len(filtered_text)
        if remaining > 0:
            filtered_text += ['zerovec']*remaining
        else:
            filtered_text = filtered_text[:max_len_p]
    else:
        filtered_text = text.lower().split()
        if remaining > 0:
            filtered_text += ['zerovec']*remaining
        else:
            filtered_text = filtered_text[:max_len_q]

    processed = ["".join(list(filter(str.isalnum, text))) for text in filtered_text]
    # singularize = [p.stem(word) for word in processed]
    vector_array = []

    for i, word in enumerate(processed):
        if word in GloveEmbeddings:
            vector_array.append([float(x) for x in GloveEmbeddings[word]])
        else:
            vector_array.append(GloveEmbeddings['zerovec'])

    return vector_array

