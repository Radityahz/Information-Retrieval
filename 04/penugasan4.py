#Import modul
import os
import re
import numpy as np
import pandas as pd  
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
    # regex menghapus karakter yang bukan alfabet
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Memecah teks menjadi token berdasarkan spasi
    return text.split()

#Folder Path
path = r"E:\\KULIAH\\TINGKAT 3\\IR\\PRAKTIKUM\\04\\berita"

# Mengambil semua file dalam folder
files = os.listdir(path)

# Inverted index
inverted_index = {}
doc_names = {}  # Menyimpan nama dokumen berdasarkan ID
doc_dict = {}  # Menyimpan konten dokumen untuk edit distance

#Preprocessing dan Inverted Index
for doc_id, filename in enumerate(files, start=1):
    # Memastikan hanya memproses file teks
    if filename.endswith('.txt'):
        file_path = os.path.join(path, filename)
        doc_names[doc_id] = filename  # Menyimpan nama file berdasarkan ID
        
        # Membaca konten file
        with open(file_path, 'r') as file:
            dokumen = file.read()

            # Case-folding
            casefolded_dokumen = dokumen.lower()

            # Tokenisasi
            tokens = tokenize(casefolded_dokumen)

            # Eliminasi stopwords
            filtered_tokens = [token for token in tokens if token not in stopwords]

            # Stemming
            stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

            # Inverted Index Construction
            for pos, term in enumerate(stemmed_tokens):  
                if term not in inverted_index:
                    inverted_index[term] = {}
                if doc_id not in inverted_index[term]:
                    inverted_index[term][doc_id] = []
                inverted_index[term][doc_id].append(pos)

            # Menyimpan konten untuk edit distance
            doc_dict[filename] = dokumen

# Daftar vocabulary berdasarkan inverted index
vocab = list(inverted_index.keys())

#Fungsi menghitung document frequncy (DF)
def wordDocFre(vocab, inverted_index, doc_names):
    df = {}
    for doc_id, doc_name in doc_names.items():
        df[doc_name] = {}
        for word in vocab:
            if doc_id in inverted_index.get(word, {}):
                df[doc_name][word] = len(inverted_index[word][doc_id])
            else:
                df[doc_name][word] = 0
    return df
#Menghitung DF
df_result = wordDocFre(vocab, inverted_index, doc_names)

#Fungsi inverse document frequency (IDF)
def inverseDocFre(vocab, doc_fre, length):
    idf = {}
    for word in vocab:
        doc_freq = sum(1 for doc in doc_fre.values() if doc[word] > 0) #hitung dokumen yang memiliki term
        idf[word] = 1 + np.log((length + 1) / (doc_freq + 1)) if doc_freq > 0 else 0 
    return idf
# Menghitung IDF
idf_result = inverseDocFre(vocab, df_result, len(doc_names))

#Fungsi menghitung Term Frequency
def termFrequencyInDoc(vocab, doc_fre):
    tf = {}
    for doc_name, freqs in doc_fre.items():
        tf[doc_name] = {}
        total_terms = sum(freqs.values())
        for word in vocab:
            tf[doc_name][word]= freqs[word] / total_terms if total_terms > 0 else 0
    return tf
#Menghitung TF
tf_result = termFrequencyInDoc(vocab, df_result)

#Fungsi dictionary tf.idf
def tfidf(vocab,tf,idf_scr):
    tf_idf_scr = {}
    for doc_name in tf.keys():
        tf_idf_scr[doc_name] = {}
    for word in vocab:
        tf_idf_scr[doc_name][word] = tf[doc_name][word] * idf_scr[word]
    return tf_idf_scr
#Panggil Fungsi tfidf
tf_idf = tfidf(vocab, tf_result, idf_result)

#Term - Document Matrix
TD = np.zeros((len(vocab), len(doc_names)))
for i, word in enumerate(vocab):
    for j, doc_name in enumerate(doc_names.values()):
        # Periksa apakah dokumen dan term ada di tf_idf
        if doc_name in tf_idf and word in tf_idf[doc_name]:
            TD[i][j] = tf_idf[doc_name][word]
        else:
            TD[i][j] = 0
# Menggunakan pandas untuk menampilkan hasil matriks
td_df = pd.DataFrame(TD, index=vocab, columns=doc_names.values())

# Tampilkan hasil
print("Term - Document Matrix (TD): ")
print(td_df)

#Edit Distance
def edit_distance(str1, str2):
    m = len(str1)
    n = len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1])
    return dp[m][n]

# Jaccard Similarity
def jaccard_sim(list1, list2):
    intersection = len(set(list1).intersection(set(list2)))
    union = len(set(list1).union(set(list2)))
    return float(intersection) / union if union != 0 else 0

#Euclidean Distance
def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((np.array(vec1) - np.array(vec2))**2))

# Cosine Similarity
def cosine_sim(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 != 0 else 0

# Fungsi untuk menghitung similarity antara semua dokumen
def compare_documents(doc1_name, doc2_name, td_df):
    vec1 = td_df[doc1_name].values
    vec2 = td_df[doc2_name].values

    # Jaccard similarity
    jaccard_similarity = jaccard_sim(vec1, vec2)
    
    # Euclidean distance
    euclidean_distance = euclidean_dist(vec1, vec2)
    
    # Cosine similarity
    cosine_similarity = cosine_sim(vec1, vec2)

    return jaccard_similarity, euclidean_distance, cosine_similarity

def compare_all_documents(td_df):
    results = []

    # Mengambil semua nama dokumen
    doc_names_list = list(td_df.columns)
    
    # Iterasi melalui setiap pasangan dokumen
    for i in range(len(doc_names_list)):
        for j in range(i + 1, len(doc_names_list)):  # Hanya periksa pasangan unik
            doc1_name = doc_names_list[i]
            doc2_name = doc_names_list[j]
            
            # Menghitung similarity
            jaccard_similarity, euclidean_distance, cosine_similarity = compare_documents(doc1_name, doc2_name, td_df)
            
            # Menghitung Edit Distance antara nama dokumen
            edit_distance_value = edit_distance(doc_dict[doc1_name], doc_dict[doc2_name])

            # Menyimpan hasil ke dalam list
            results.append({
                'Document 1': doc1_name,
                'Document 2': doc2_name,
                'Edit Distance': edit_distance_value,
                'Jaccard Similarity': jaccard_similarity,
                'Euclidean Distance': euclidean_distance,
                'Cosine Similarity': cosine_similarity
            })
    
    # Mengubah hasil ke dalam DataFrame untuk tampilan yang lebih baik
    results_df = pd.DataFrame(results)
    return results_df

# Panggil fungsi untuk menghitung similarity antara semua dokumen
similarity_results = compare_all_documents(td_df)

# Tampilkan hasil
print("Hasil Similarity antara Semua Dokumen:")
print(similarity_results)
