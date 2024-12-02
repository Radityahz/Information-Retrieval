def tokenisasi(text):
    tokens = text.split(" ") 
    return tokens 

def stemming(text):
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory 
    # create stemmer 
    factory = StemmerFactory() 
    stemmer = factory.create_stemmer() 
    # stemming process 
    output   = stemmer.stem(text) 
    return output 

def stemming_sentence(text):
    output = "" 
    for token in tokenisasi(text): 
        output   = output + stemming(token) + " " 
    return output[:-1] 

doc_dict_raw = {}
doc_dict_raw['doc1'] = "pengembangan sistem informasi penjadwalan"
doc_dict_raw['doc2'] = "pengembangan model analisis sentimen berita"
doc_dict_raw['doc3'] = "analisis sistem input output"
doc_dict_raw['doc4'] = "pengembangan sistem informasi akademik universitas"
doc_dict_raw['doc5'] = "pengembangan sistem cari berita ekonomi"
doc_dict_raw['doc6'] = "analisis sistem neraca nasional"
doc_dict_raw['doc7'] = "pengembangan sistem informasi layanan statistik"
doc_dict_raw['doc8'] = "pengembangan sistem pencarian skripsi di universitas"
doc_dict_raw['doc9'] = "analisis sentimen publik terhadap pemerintah"
doc_dict_raw['doc10'] = "pengembangan model klasifikasi sentimen berita"

doc_dict = {}
for doc_id,doc in doc_dict_raw.items():
    doc_dict[doc_id] = stemming_sentence(doc) 
print(doc_dict) 

vocab = []
inverted_index = {}
for doc_id,doc in doc_dict.items():
    for token in tokenisasi(doc): 
        print(token) 
        if token not in vocab: 
            vocab.append(token) 
            inverted_index[token] = [] 
        if token in inverted_index:  
            if doc_id not in inverted_index[token]:  
                inverted_index[token].append(doc_id) 
print("Vocab: ", vocab, "\n")
print("Inverted Index: ", inverted_index) 

query = "sistem informasi statistik"
def termFrequency(vocab, query):
    tf_query = {} 
    for word in vocab: 
        tf_query[word] = query.count(word) 
    return tf_query 
tf_query = termFrequency(vocab, query)

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
doc_frequencies = wordDocFre(vocab, doc_dict)
print("Document Frequency (DF): ")
print(doc_frequencies)

#Fungsi inverse document frequency
import numpy as np
def inverseDocFre(vocab, doc_fre, length):
    idf = {}
    for word in vocab:
        idf[word] = 1 + np.log((length + 1) / (doc_fre[word] + 1))
    return idf
idf_scores = inverseDocFre(vocab, doc_frequencies, len(doc_dict))
print("Inverse Document Frequency (IDF): ")
print(idf_scores)
print()

#Fungsi dictionary tf.idf
def tfidf(vocab, doc_dict, idf_scores):
    tf_idf_scr = {}
    for doc_id in doc_dict.keys():
        tf_idf_scr[doc_id] = {}
        for word in vocab:
            tf_idf_scr[doc_id][word] = doc_dict[doc_id].count(word) * idf_scores[word]  # Menggunakan DF untuk menghitung TF
    return tf_idf_scr

#Panggil Fungsi tfidf
tf_idf = tfidf(vocab, doc_dict, idf_scores)

# Term - Query Matrix
TQ = np.zeros((len(vocab), 1)) #hanya 1 query
for word in vocab:
    ind1 = vocab.index(word) 
    TQ[ind1][0] = tf_query[word]*idf_scores[word] 
print("Term - Query Matrix: ")
print(TQ) 

#Term - Document Matrix
TD = np.zeros((len(vocab), len(doc_dict)))
for word in vocab:
    for doc_id, doc in tf_idf.items():
        ind1 = vocab.index(word)
        ind2 = list(tf_idf.keys()).index(doc_id)
        TD[ind1][ind2] = tf_idf[doc_id][word]
print("Term - Document Matrix: ")
print(TD)

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

# Menghitung Similarity
print("Cosine Similarity:")
for i, doc_id in enumerate(doc_dict.keys()):
    sim = cosine_sim(TQ[:, 0], TD[:, i])
    print(f"Query & {doc_id}: {sim}")
print()

#Mengambil k skor untuk suatu query
from collections import OrderedDict
def exact_top_k(doc_dict, TD, q, k):
    relevance_scores = {} 
    i = 0 
    for doc_id in doc_dict.keys(): 
        relevance_scores[doc_id] = cosine_sim(q, TD[:, i]) 
        i = i + 1 
    sorted_value = OrderedDict(sorted(relevance_scores.items(), 
key=lambda x: x[1], reverse = True))
    top_k = {j: sorted_value[j] for j in list(sorted_value)[:k]} 
    return top_k 
top_2 = exact_top_k(doc_dict, TD, TQ[:, 0], 2) 
#Misalkan k=3, maka:
top_3 = exact_top_k(doc_dict, TD, TQ[:, 0], 3)
print("Ketika k=2, maka: ", top_2)
print("Ketika k=3, maka: ", top_3)

#Index Elimination
def index_elim_simple(query, doc_dict):
    remove_list = []
    for doc_id, doc in doc_dict.items():
        n = 0
        for word in tokenisasi(query):
            if stemming(word) in doc:
                n = n+1
        if n==0:
            remove_list.append(doc_id)
    for key in remove_list:
        del doc_dict[key]
    return doc_dict
query = "sistem informasi statistik"
doc_dict = index_elim_simple(query, doc_dict)
print("Dokumen hasil eliminasi: ", doc_dict)

#Eliminasi dengan menggunakan nilai idf
def elim_query(query, idf_dict, idf_score):
    for term in tokenisasi(query):
        if idf_dict[stemming(term)]<idf_score:
            query = query.replace(term + " ", "")
            query = query.replace(term, "")
    return query
#Misal menggunakan idf_score=15
query = "sistem informasi statistik"
query = elim_query(query, idf_scores, 1.5)
print(query)

#Champion List
def create_championlist(inverted_index, tf_idf, r):
    champion_list = {}
    for term in inverted_index.keys():
        weight_scores = {}
        for doc_id, tf in tf_idf.items():
            if tf_idf[doc_id][term]!=0:
                weight_scores[doc_id]= tf_idf[doc_id][term]
        sorted_value = OrderedDict(sorted(weight_scores.items(), key=lambda x: x[1], reverse = True))
        top_r = {j: sorted_value[j] for j in list(sorted_value)[:r]}
        champion_list[term]=list(top_r.keys())
    return champion_list
r=2
print("Champion list untuk r=2, ", create_championlist(inverted_index,tf_idf,r))
print()

#Evaluasi untuk Unranked Retrieval Set
top_3 = {'doc7': 0.7689768599816609, 'doc1': 0.4641504133851462, 'doc4': 0.35626622628022314}
rel_judgement1 = {'doc1':1, 'doc2':0, 'doc3':0, 'doc4':1, 'doc5':1, 'doc6':0, 'doc7':1, 'doc8':1, 'doc9':0, 'doc10':0}
rel_docs = []
for doc_id, rel in rel_judgement1.items():
    if rel==1:
        rel_docs.append(doc_id)

retrieved_rel_doc3 = [value for value in list(top_3.keys()) if value in rel_docs]
prec3 = len(retrieved_rel_doc3)/len(top_3)*100
rec3 = len(retrieved_rel_doc3)/len(rel_docs)*100
fScore3 = 2 * prec3 * rec3 / (prec3 + rec3)
print("Precision 3 Dokumen Teratas: ", prec3) 
print("Recall 3 Dokumen Teratas: ", rec3)
print("F-measure 3 Dokumen Teratas: ", fScore3)
print()

#untuk 5 dokumen teratas
top_5 = {'doc7': 0.7689768599816609, 'doc1': 0.4641504133851462, 'doc4': 0.35626622628022314, 'doc3': 0.10856998991379904, 'doc6': 0.10856998991379904}
rel_judgement1 = {'doc1':1, 'doc2':0, 'doc3':0, 'doc4':1, 'doc5':1, 'doc6':0, 'doc7':1, 'doc8':1, 'doc9':0, 'doc10':0}
rel_docs = []
for doc_id, rel in rel_judgement1.items(): 
    if rel==1: 
        rel_docs.append(doc_id) 
retrieved_rel_doc5 = [value for value in list(top_5.keys()) if value in rel_docs]
prec5 = len(retrieved_rel_doc5)/len(top_5)*100
rec5 = len(retrieved_rel_doc5)/len(rel_docs)*100
fScore5 = 2 * prec5 * rec5 / (prec5 + rec5)
print("Precision 5 Dokumen Teratas: ", prec5) 
print("Recall 5 Dokumen Teratas: ", rec5)
print("F-measure 5 Dokumen Teratas: ", fScore5)

#Evaluasi untuk Ranked Retrieval Set
def compute_prf_metrics(I, score, I_Q):
    """Compute precision, recall, F-measures and other
    evaluation metrics for document-level retrieval

    Args:
        I (np.ndarray): Array of items
        score (np.ndarray): Array containing the score values of the times
        I_Q (np.ndarray): Array of relevant (positive) items

    Returns:
        P_Q (float): Precision
        R_Q (float): Recall
        F_Q (float): F-measures sorted by rank
        BEP (float): Break-even point
        F_max (float): Maximal F-measure
        P_average (float): Mean average
        X_Q (np.ndarray): Relevance function
        rank (np.ndarray): Array of rank values
        I_sorted (np.ndarray): Array of items sorted by rank
        rank_sorted (np.ndarray): Array of rank values sorted by rank
    """
    # Compute rank and sort documents according to rank
    K = len(I)
    index_sorted = np.flip(np.argsort(score))
    I_sorted = I[index_sorted]
    rank = np.argsort(index_sorted) + 1
    rank_sorted = np.arange(1, K+1)

    # Compute relevance function X_Q (indexing starts with zero)
    X_Q = np.isin(I_sorted, I_Q)


    # Compute precision and recall values (indexing starts with zero)
    M = len(I_Q)
    P_Q = np.cumsum(X_Q) / np.arange(1, K+1)
    R_Q = np.cumsum(X_Q) / M

    # Break-even point
    BEP = P_Q[M-1]
    # Maximal F-measure
    sum_PR = P_Q + R_Q
    sum_PR[sum_PR == 0] = 1  # Avoid division by zero
    F_Q = 2 * (P_Q * R_Q) / sum_PR
    F_max = F_Q.max()
    # Average precision
    P_average = np.sum(P_Q * X_Q) / len(I_Q)

    return P_Q, R_Q, F_Q, BEP, F_max, P_average, X_Q, rank, I_sorted, rank_sorted 

#Panggil Fungsi
relevance_score1 = {'doc1': 0.4641504133851462, 'doc2': 0.0,
'doc3': 0.10856998991379904, 'doc4': 0.35626622628022314,
'doc5': 0.10705617011820337, 'doc6': 0.10856998991379904,
'doc7': 0.7689768599816609, 'doc8': 0.08967792817935699,
'doc9': 0.0, 'doc10': 0.0}
I = np.array(list(relevance_score1.keys()))
score = np.array(list(relevance_score1.values()))
I_Q = np.array(['doc1', 'doc4', 'doc5', 'doc7', 'doc8'])
output = compute_prf_metrics(I, score, I_Q)
P_Q, R_Q, F_Q, BEP, F_max, P_average, X_Q, rank, I_sorted, rank_sorted = output

# Arrange output as tables
import pandas as pd
score_sorted = np.flip(np.sort(score))
df = pd.DataFrame({'Rank': rank_sorted, 'ID': I_sorted,
                   'Score': score_sorted,
                   '$\chi_\mathcal{Q}$': X_Q, 
                   'P(r)': P_Q, 
                   'R(r)': R_Q,
                   'F(r)': F_Q})
print(df)

print('Break-even point = %.2f' % BEP)
print('F_max = %.2f' % F_max)
print('Average precision =', np.round(P_average, 5)) 

#Fungsi Membuat Kurva Precision-recall
from matplotlib import pyplot as plt

def plot_PR_curve(P_Q, R_Q, figsize=(3, 3)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.plot(R_Q, P_Q, linestyle='--', marker='o', color='k', mfc='r')
    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1])
    ax.set_aspect('equal', 'box')
    plt.title('PR curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid()
    plt.tight_layout()
    ax.plot(BEP, BEP, color='green', marker='o',
fillstyle='none', markersize=15)
    ax.set_title('PR curve')
    #plt.show()
    return fig, ax 

# plot_PR_curve(P_Q, R_Q, figsize=(3,3)) 

from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

DT = TD.transpose()
#print(DT)
model = TruncatedSVD(n_components=2, random_state=7).fit(DT)
DT_reduced = model.transform(DT)
QT_reduced = model.transform(TQ.transpose())
print(QT_reduced)
print(DT_reduced)
plt.scatter(DT_reduced[:, 0], DT_reduced[:, 1])
plt.scatter(QT_reduced[:, 0], QT_reduced[:, 1], color=["red"])
labels=list(doc_dict.keys())
for i, txt in enumerate(labels):
    plt.annotate(txt, (DT_reduced[i, 0], DT_reduced[i, 1])) 
plt.annotate("query", (QT_reduced[0, 0], QT_reduced[0, 1]))
#plt.show() 

top_5 = exact_top_k(doc_dict, TD, TQ[:, 0], 5)
print(top_5) 

rel_vecs_id = ["doc1", "doc4", "doc5", "doc7", "doc8"]
nrel_vecs_id = ["doc2", "doc3", "doc6", "doc9", "doc10"]

doc_ids = ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6","doc7", "doc8", "doc9", "doc10"]

rel_vecs = []
for doc in rel_vecs_id:
    rel_vecs.append(DT[doc_ids.index(doc),:]) 

nrel_vecs = []
for doc in nrel_vecs_id:
    nrel_vecs.append(DT[doc_ids.index(doc),:]) 

query_vecs = TQ.transpose()
alpha = 1
beta = 0.75
gamma = 0.15

# Update query vectors with Rocchio algorithm
query_vecs = alpha * query_vecs + beta * np.mean(rel_vecs, axis=0) - gamma * np.mean(nrel_vecs, axis=0)
query_vecs[query_vecs<0] = 0 #negative value => 0 

top_5 = exact_top_k(doc_dict, TD, query_vecs[0, :].transpose(), 5)
print(top_5) 

QT1_reduced = model.transform(query_vecs)

plt.scatter(DT_reduced[:, 0], DT_reduced[:, 1])
plt.scatter(QT_reduced[:, 0], QT_reduced[:, 1], color=["red"])
plt.scatter(QT1_reduced[:, 0], QT1_reduced[:, 1], color=["green"])
doc_ids=list(doc_dict.keys())
for i, txt in enumerate(doc_ids):
    plt.annotate(txt, (DT_reduced[i, 0], DT_reduced[i, 1])) 
plt.annotate("query", (QT_reduced[0, 0], QT_reduced[0, 1]))
plt.annotate("new query", (QT1_reduced[:, 0], QT1_reduced[:, 1]))
plt.show() 

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4') 

from itertools import chain
from nltk.corpus import wordnet
query = "information system"

expand_list = []
for term in query.split(" "):
    synonyms = wordnet.synsets(term) 
    lemmas = set(chain.from_iterable([word.lemma_names() for word in synonyms]))
 
print(lemmas) 
 
expand_list = expand_list + list(lemmas) 
print(expand_list)

query_expand = query + " " + (" ".join(expand_list)).replace("_", " ")
print(query_expand) 