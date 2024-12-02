import numpy as np
import math
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def tokenisasi(text):
    return text.split(" ")

def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return stemmer.stem(text)

def stemming_sentence(text):
    return " ".join(stemming(token) for token in tokenisasi(text))

# Dokumen mentah
doc_dict_raw = {
    'doc1': "pengembangan sistem informasi penjadwalan",
    'doc2': "pengembangan model analisis sentimen berita",
    'doc3': "analisis sistem input output",
    'doc4': "pengembangan sistem informasi akademik universitas",
    'doc5': "pengembangan sistem cari berita ekonomi",
    'doc6': "analis sistem neraca nasional",
    'doc7': "pengembangan sistem informasi layanan statistik",
    'doc8': "pengembangan sistem pencarian skripsi di universitas",
    'doc9': "analisis sentimen publik terhadap pemerintah",
    'doc10': "pengembangan model klasifikasi sentimen berita"
}

# Stemming dokumen
doc_dict = {doc_id: stemming_sentence(doc) for doc_id, doc in doc_dict_raw.items()}

# Membuat vocab dan inverted index
vocab = []
inverted_index = {}
for doc_id, doc in doc_dict.items():
    for token in tokenisasi(doc):
        if token not in vocab:
            vocab.append(token)
            inverted_index[token] = []
        if doc_id not in inverted_index[token]:
            inverted_index[token].append(doc_id)

print("Vocab: ", vocab)
print("Inverted Index:", inverted_index)

query = "sistem informasi statistik"

# Fungsi menghitung frekuensi term dalam dokumen
def termFrequencyInDoc(vocab, doc_dict):
    tf_docs = {}
    for doc_id in doc_dict.keys():
        tf_docs[doc_id] = {}
        for word in vocab:
            tf_docs[doc_id][word] = doc_dict[doc_id].split().count(word)
    return tf_docs

def termFrequency(vocab, query):
    tf_query = {}
    for word in vocab:
        tf_query[word] = query.split().count(word)
    return tf_query

tf_query = termFrequency(vocab, query)

# Menghitung DF
def wordDocFre(vocab, doc_dict):
    df = {}
    for word in vocab:
        frq = sum(1 for doc in doc_dict.values() if word in tokenisasi(doc))
        df[word] = frq
    return df

doc_frequencies = wordDocFre(vocab, doc_dict)
print("Document Frequency (DF): ")
print(doc_frequencies)

# Menghitung IDF
def inverseDocFre(vocab, doc_fre, length):
    idf = {}
    for word in vocab:
        idf[word] = 1 + np.log((length + 1) / (doc_fre[word] + 1))
    return idf

idf_scores = inverseDocFre(vocab, doc_frequencies, len(doc_dict))
print("Inverse Document Frequency (IDF): ")
print(idf_scores)

# Fungsi dictionary TF-IDF
def tfidf(vocab, tf, idf_scr):
    tf_idf_scr = {}
    for doc_id in tf.keys():
        tf_idf_scr[doc_id] = {}
        for word in vocab:
            tf_idf_scr[doc_id][word] = tf[doc_id][word] * idf_scr[word]
    return tf_idf_scr

# Panggil Fungsi TF-IDF
tf_idf = tfidf(vocab, termFrequencyInDoc(vocab, doc_dict), idf_scores)

# Term - Document Matrix
TD = np.zeros((len(vocab), len(doc_dict)))
for i, word in enumerate(vocab):
    for j, doc_id in enumerate(doc_dict.keys()):
        TD[i][j] = tf_idf[doc_id].get(word, 0)

print("Term-Document Matrix (TD):")
print(TD)

# Cosine Similarity
def cosine_sim(vec1, vec2):
    dot_prod = np.dot(vec1, vec2)
    mag_1 = np.linalg.norm(vec1)
    mag_2 = np.linalg.norm(vec2)
    if mag_1 == 0 or mag_2 == 0:
        return 0.0  # Menghindari pembagian dengan nol
    return dot_prod / (mag_1 * mag_2)

# Matriks Term-Query (TQ)
TQ = np.zeros((len(vocab), 1))  # hanya 1 query
for i, word in enumerate(vocab):
    TQ[i][0] = tf_query[word] * idf_scores[word]

print("Matriks Term-Query (TQ):")
print(TQ)

# Menghitung Similaritas Cosinus antara query dan dokumen
for j, doc_id in enumerate(doc_dict.keys()):
    sim_score = cosine_sim(TQ[:, 0], TD[:, j])
    print(f"Similaritas antara query dan {doc_id}: {sim_score}")
