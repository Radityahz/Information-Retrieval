from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

#Fungsi menghitung frekuensi term dalam dokumen
def termFrequencyInDoc(vocab, doc_dict):
    tf_docs = {}
    for doc_id in doc_dict.keys():
        tf_docs[doc_id] = {}
        for word in vocab:
            tf_docs[doc_id][word] = doc_dict[doc_id].count(word)
    return tf_docs

#create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

doc1_term = ["pengembangan", "sistem", "informasi", "penjadwalan"]
doc2_term = ["pengembangan", "model", "analisis", "sentimen", "berita"]
doc3_term = ["analisis", "sistem", "input", "output"]

corpus_term = [doc1_term, doc2_term, doc3_term]

#Inverted Index
inverted_index = {}

for i in range(len(corpus_term)):
    for item in corpus_term[i]:
        item = stemmer.stem(item)
        if item not in inverted_index:
            inverted_index[item] = []
        if (i+1) not in inverted_index[item]:
            inverted_index[item].append(i+1)

vocab=list(inverted_index.keys())
doc_dict = {}
#clean after stemming
doc_dict['doc1'] = "kembang sistem informasi jadwal"
doc_dict['doc2'] = "kembang model analisis sentimen berita"
doc_dict['doc3'] = "analisis sistem input output"

print(termFrequencyInDoc(vocab, doc_dict), "\n")

def tokenisasi(doc):
    return doc.split()

#Fungsi menghitung document frequncy
def wordDocFre(vocab, doc_dict):
    df = {}
    for word in vocab:
        frq = 0
        for doc in doc_dict.values():
            if word in tokenisasi(doc):
                frq = frq + 1
        df[word] = frq
    return df
print("Document Frequency (DF): ")
print(wordDocFre(vocab, doc_dict))
print()

#Fungsi inverse document frequency
import numpy as np
def inverseDocFre(vocab, doc_fre, length):
    idf = {}
    for word in vocab:
        idf[word] = 1 + np.log((length + 1) / (doc_fre[word] + 1))
    return idf
print("Inverse Document Frequency (IDF): ")
print(inverseDocFre(vocab, wordDocFre(vocab, doc_dict), len(doc_dict)))
print()

#Fungsi dictionary tf.idf
def tfidf(vocab,tf,idf_scr,doc_dict):
    tf_idf_scr = {}
    for doc_id in doc_dict.keys():
        tf_idf_scr[doc_id] = {}
    for word in vocab:
        for doc_id,doc in doc_dict.items():
            tf_idf_scr[doc_id][word] = tf[doc_id][word] * idf_scr[word]
    return tf_idf_scr

#Panggil Fungsi tfidf
tf_idf = tfidf(vocab, termFrequencyInDoc(vocab, doc_dict), inverseDocFre(vocab,wordDocFre(vocab, doc_dict), len(doc_dict)), doc_dict)

#Term - Document Matrix
TD = np.zeros((len(vocab), len(doc_dict)))
for word in vocab:
    for doc_id, doc in tf_idf.items():
        ind1 = vocab.index(word)
        ind2 = list(tf_idf.keys()).index(doc_id)
        TD[ind1][ind2] = tf_idf[doc_id][word]
print(TD)
print()

#1. Edit Distance
def edit_distance(string1, string2):
    if len(string1) > len(string2):
        difference = len(string1) - len(string2)
        string1[:difference]
        n = len(string2)
    elif len(string2) > len(string1):
        difference = len(string2) - len(string1)
        string2[:difference]
        n = len(string1)
    for i in range(n):
        if string1[i] != string2[i]:
            difference += 1
    return difference
print(edit_distance(doc_dict['doc1'], doc_dict['doc2']))
print(edit_distance(doc_dict['doc1'], doc_dict['doc3']))
print()

#Jaccard Similarity
def jaccard_sim(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union
print(jaccard_sim(doc_dict['doc1'].split(" "), doc_dict['doc2'].split(" ")))
print(jaccard_sim(doc_dict['doc1'].split(" "), doc_dict['doc3'].split(" ")))
print()

#Euclidian Distance
def euclidian_dist(vec1, vec2):
    #substracting vector
    temp = vec1 - vec2

    #doing dot product
    #for finding
    #sum of the square
    sum_sq = np.dot(temp.T, temp)

    #Doing squareroot and 
    #printing Euclidean distance
    return np.sqrt(sum_sq)
print(euclidian_dist(TD[:, 0], TD[:, 1])) #doc1 & doc2
print(euclidian_dist(TD[:, 0], TD[:, 2])) #doc1 & doc3
print()

#Cosine Similarity
import math
def cosine_sim(vec1, vec2):
    vec1 = list(vec1)
    vec2 = list(vec2)
    dot_prod = 0
    for i, v in enumerate(vec1):
        dot_prod += v * vec2[i]
    mag_1 = math.sqrt(sum([x**2 for x in vec1]))
    mag_2 = math.sqrt(sum([x**2 for x in vec2]))
    return dot_prod / (mag_1 * mag_2)
print(cosine_sim(TD[:, 0], TD[:, 1])) #doc1 & doc2
print(cosine_sim(TD[:, 0], TD[:, 2])) #doc1 & doc3
print()

query = "sistem informasi statistik"
def termFrequency(vocab, query):
    tf_query = {}
    for word in vocab:
        tf_query[word] = query.split().count(word)
    return tf_query

tf_query = termFrequency(vocab, query)

#Term - Query Matrix
TQ = np.zeros((len(vocab), 1)) #hanya 1 query
for word in vocab:
    ind1 = vocab.index(word)
    TQ[ind1][0] = tf_query[word] * tf_idf[doc_id][word]
print("Matriks TQ")
print(TQ)

print("Similaritas Cosinus Query & Dokumen:")
for doc_id in doc_dict.keys():
    doc_index = list(doc_dict.keys()).index(doc_id)
    print(f"Similaritas antara query dan {doc_id}: {cosine_sim(TQ[:, 0], TD[:, doc_index])}")

from collections import OrderedDict
def exact_top_k(doc_dict, TD, q, k):
    relevance_scores = {}
    i = 0
    for doc_id in doc_dict.keys():
        relevance_scores[doc_id] = cosine_sim(q, TD[:, i])
        i = i+1
    sorted_value = OrderedDict(sorted(relevance_scores.items(), key=lambda x: x[1], reverse = True))
    top_k = {j: sorted_value[j] for j in list(sorted_value)[:k]}
    return top_k
top_2 = exact_top_k(doc_dict, TD, TQ[:, 0], 2)
#Misalkan k=3 maka:
top_3 = exact_top_k(doc_dict, TD, TQ[:, 0], 3)