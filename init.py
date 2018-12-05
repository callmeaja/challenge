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
        remaining = max_len_p - len(filtered_text)
        if remaining > 0:
            filtered_text += ['zerovec']*remaining
        else:
            filtered_text = filtered_text[:max_len_p]
    else:
        filtered_text = text.lower().split()
        remaining = max_len_q - len(filtered_text)
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


def batch_iter(data, batch_size, num_epochs, shuffle=False):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]