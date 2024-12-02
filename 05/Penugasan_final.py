import os
import re
import numpy as np
import pandas as pd
from collections import OrderedDict
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Inisialisasi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi untuk tokenisasi
def tokenisasi(text):
    tokens = text.split(" ")
    return tokens

# Fungsi untuk melakukan stemming
def stemming(text):
    output = stemmer.stem(text)
    return output

# Fungsi untuk stemming satu kalimat
def stemming_sentence(text):
    output = ""
    for token in tokenisasi(text):
        output += stemming(token) + " "
    return output.strip()

# Fungsi untuk membaca dokumen dari folder
def read_documents(folder_path):
    doc_dict_raw = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Anggap file dokumen memiliki ekstensi .txt
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                doc_dict_raw[filename] = file.read().strip()
    return doc_dict_raw

# Fungsi untuk menghitung Term Frequency (TF) dari query
def termFrequency(vocab, query):
    tf_query = {}
    for word in vocab:
        tf_query[word] = query.count(word)
    return tf_query

# Fungsi cosine similarity
def cosine_sim(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2) if norm_vec1 * norm_vec2 != 0 else 0

# Fungsi untuk mendapatkan top k dokumen yang paling relevan
def exact_top_k(doc_dict, TD, q, k):
    relevance_scores = {}
    for i, doc_id in enumerate(doc_dict.keys()):
        relevance_scores[doc_id] = cosine_sim(q, TD[:, i])
    # Mengurutkan berdasarkan skor tertinggi
    sorted_scores = OrderedDict(sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True))
    top_k = {doc: score for doc, score in list(sorted_scores.items())[:k]}
    return top_k

# Fungsi untuk mendapatkan semua skor relevansi
def all_scores(doc_dict, TD, q):
    relevance_scores = {}
    for i, doc_id in enumerate(doc_dict.keys()):
        relevance_scores[doc_id] = cosine_sim(q, TD[:, i])
    return relevance_scores

# Fungsi main
def main():
    folder_path = r"E:\\KULIAH\\TINGKAT 3\\IR\\PRAKTIKUM\\04\\berita"  # Ganti dengan path folder dokumen berita
    query = "vaksin corona jakarta"

    # Membaca dokumen dari folder
    doc_dict_raw = read_documents(folder_path)

    # Melakukan stemming pada setiap dokumen
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

    # Menghitung IDF
    N = len(doc_dict)
    idf = {}
    for word in vocab:
        df = len(inverted_index[word])  # jumlah dokumen yang mengandung kata
        idf[word] = np.log((N / (df + 1)))  # Tambahkan 1 untuk mencegah pembagian nol

    # Memproses query dengan stemming dan tokenisasi
    query_stemmed = stemming_sentence(query)

    # Menghitung TF untuk query
    tf_query = termFrequency(vocab, query_stemmed)

    # Membuat Term-Query Matrix
    TQ = np.zeros((len(vocab), 1))  # hanya 1 query
    for word in vocab:
        ind1 = vocab.index(word)
        TQ[ind1][0] = tf_query[word] * idf.get(word, 0)  # kalikan dengan idf

    # Membuat Matriks Term-Document (TD)
    TD = np.zeros((len(vocab), len(doc_dict)))  # jumlah term x jumlah dokumen
    for i, doc in enumerate(doc_dict.values()):
        tf_doc = termFrequency(vocab, doc)
        for word in vocab:
            TD[vocab.index(word), i] = tf_doc[word] * idf.get(word, 0)

     # Mendapatkan semua skor relevansi dengan query
    all_relevance_scores = all_scores(doc_dict, TD, TQ[:, 0])

    print("All Documents Scores:")
    for doc, score in all_relevance_scores.items():
        print(f"{doc}: {score:.4f}")

    # Mendapatkan top 3 dokumen yang paling relevan dengan query
    top_3 = exact_top_k(doc_dict, TD, TQ[:, 0], 3)

    print("Top 3 Documents:")
    for doc, score in top_3.items():
        print(f"{doc}: {score:.4f}")
    
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
        return remove_list
    
    removed_docs = index_elim_simple(query, doc_dict)
    print("Dokumen hasil eliminasi:")
    for doc in removed_docs:
        print(doc)

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
    
    relevance_score1 = { 'berita1.txt': 0.0032, 'berita2.txt': 0.485, 'berita3.txt': 0.1364, 'berita4.txt': 0.0127,
    'berita5.txt': 0.0502}
    I = np.array(list(relevance_score1.keys()))
    score = np.array(list(relevance_score1.values()))
    I_Q = np.array(['berita1.txt', 'berita2.txt', 'berita3.txt', 'berita4.txt', 'berita5.txt'])
    output = compute_prf_metrics(I, score, I_Q)
    P_Q, R_Q, F_Q, BEP, F_max, P_average, X_Q, rank, I_sorted, rank_sorted = output

    # Arrange output as tables
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

# Memanggil fungsi main
if __name__ == "__main__":
    main()
