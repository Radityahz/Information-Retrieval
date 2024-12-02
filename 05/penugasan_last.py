import os
import re
import numpy as np
import pandas as pd
from collections import OrderedDict
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Daftar stopwords 
stopwords = set([
    'yang', 'dan', 'di', 'ke', 'dari', 'dengan', 'untuk', 'pada', 
    'adalah', 'ini', 'itu', 'tersebut', 'mereka', 'kami', 'kita', 
    'saya', 'anda', 'juga', 'tetapi', 'sebuah', 'sebagai', 'atas',
    'atau', 'oleh', 'itu', 'saat', 'itu', 'sebelum', 'setelah', 
    'apa', 'bagaimana', 'mana', 'mengapa', 'mungkin'
])

# Inisialisasi stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi untuk tokenisasi
def tokenize(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.split()

# Fungsi untuk membaca dokumen
def read_documents(folder_path):
    files = os.listdir(folder_path)
    doc_dict = {}
    for filename in files:
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                doc_dict[filename] = file.read()
    return doc_dict

# Fungsi menghitung document frequency (DF)
def wordDocFre(vocab, inverted_index):
    df = {word: len(inverted_index.get(word, [])) for word in vocab}
    return df

# Fungsi menghitung Term Frequency
def termFrequency(vocab, doc):
    tf = {word: 0 for word in vocab}
    tokens = tokenize(doc)
    total_terms = len(tokens)
    for token in tokens:
        if token in tf:
            tf[token] += 1
    for word in tf:
        tf[word] /= total_terms if total_terms > 0 else 1  # Normalisasi
    return tf

# Fungsi inverse document frequency (IDF)
def inverseDocFre(vocab, df, total_docs):
    idf = {}
    for word in vocab:
        idf[word] = 1 + np.log((total_docs + 1) / (df[word] + 1)) if df[word] > 0 else 0
    return idf

# Fungsi untuk membuat Term-Document Matrix
def create_term_doc_matrix(vocab, doc_dict, idf):
    TD = np.zeros((len(vocab), len(doc_dict)))  # jumlah term x jumlah dokumen
    for i, (doc_name, doc) in enumerate(doc_dict.items()):
        tf_doc = termFrequency(vocab, doc)
        for word in vocab:
            TD[vocab.index(word), i] = tf_doc[word] * idf.get(word, 0)
    return TD

# Cosine Similarity
def cosine_sim(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 != 0 else 0

# Fungsi untuk mendapatkan top k dokumen yang paling relevan
def exact_top_k(doc_dict, TD, q, k):
    relevance_scores = {}
    for i, doc_id in enumerate(doc_dict.keys()):
        relevance_scores[doc_id] = cosine_sim(q, TD[:, i])
    sorted_scores = OrderedDict(sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True))
    top_k = {doc: score for doc, score in list(sorted_scores.items())[:k]}
    return top_k

# Fungsi utama
def main():
    folder_path = r"E:\\KULIAH\\TINGKAT 3\\IR\\PRAKTIKUM\\04\\berita"  # Path folder dokumen berita
    query = "vaksin corona jakarta"

    # Membaca dokumen dari folder
    doc_dict_raw = read_documents(folder_path)

    # Tokenisasi dan stemming untuk query
    query_tokens = tokenize(stemmer.stem(query))

    # Filter dokumen berdasarkan query tokens
    filtered_doc_dict = {doc_id: doc for doc_id, doc in doc_dict_raw.items() if any(token in doc for token in query_tokens)}

    # Tokenisasi dan stemming untuk dokumen yang terfilter
    doc_dict = {doc_id: stemmer.stem(doc) for doc_id, doc in filtered_doc_dict.items()}

    # Membuat vocab dan inverted index
    vocab = []
    inverted_index = {}
    for doc_id, doc in doc_dict.items():
        tokens = tokenize(doc)
        for token in tokens:
            if token not in vocab:
                vocab.append(token)
                inverted_index[token] = []
            if doc_id not in inverted_index[token]:
                inverted_index[token].append(doc_id)

    # Menghitung DF
    df_result = wordDocFre(vocab, inverted_index)

    # Menghitung IDF
    idf_result = inverseDocFre(vocab, df_result, len(doc_dict))

    # Menghitung TF untuk query
    tf_query = termFrequency(vocab, stemmer.stem(query))

    # Membuat Term-Query Matrix
    TQ = np.zeros((len(vocab), 1))  # hanya 1 query
    for word in vocab:
        TQ[vocab.index(word)][0] = tf_query[word] * idf_result.get(word, 0)  # kalikan dengan idf

    # Membuat Matriks Term-Document (TD)
    TD = create_term_doc_matrix(vocab, doc_dict, idf_result)

    # Skor relevansi untuk semua dokumen
    relevance_scores = {}
    for i, doc_id in enumerate(doc_dict.keys()):
        relevance_scores[doc_id] = cosine_sim(TQ[:, 0], TD[:, i])

    # Menampilkan hasil untuk semua dokumen
    sorted_scores = OrderedDict(sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True))
    print("Relevance Scores for all Documents:")
    for doc, score in sorted_scores.items():
        print(f"{doc}: {score:.4f}")
    
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
    
    relevance_score1 = { 'berita1.txt': 0.0, 'berita2.txt': 0.1532, 'berita3.txt': 0.2595, 'berita4.txt': 0.0,
    'berita5.txt': 0.0726}
    I = np.array(list(relevance_score1.keys()))
    score = np.array(list(relevance_score1.values()))
    I_Q = np.array(['berita2.txt', 'berita3.txt', 'berita5.txt'])
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