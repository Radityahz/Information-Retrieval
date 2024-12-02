import os
import re
import numpy as np
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

# Fungsi main
def main():
    folder_path = r"E:\\KULIAH\\TINGKAT 3\\IR\\PRAKTIKUM\\04\\berita"  # Path folder dokumen berita
    query = "vaksin corona jakarta"

    # Membaca dokumen dari folder
    doc_dict_raw = read_documents(folder_path)

    # Tokenisasi dan stemming untuk semua dokumen
    doc_dict = {doc_id: stemming_sentence(doc) for doc_id, doc in doc_dict_raw.items()}

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
    tf_query = termFrequency(vocab, stemming_sentence(query))

    # Membuat Term-Query Matrix
    TQ = np.zeros((len(vocab), 1))  # hanya 1 query
    for word in vocab:
        TQ[vocab.index(word)][0] = tf_query[word] * idf_result.get(word, 0)  

    # Membuat Matriks Term-Document (TD)
    TD = create_term_doc_matrix(vocab, doc_dict, idf_result)

    # Mendapatkan top 3 dokumen yang paling relevan dengan query
    top_3 = exact_top_k(doc_dict, TD, TQ[:, 0], 3)

    print("Top 3 Documents:")
    for doc, score in top_3.items():
        print(f"{doc}: {score:.4f}")

# Memanggil fungsi main
if __name__ == "__main__":
    main()
