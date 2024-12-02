from rank_bm25 import BM25Okapi

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

tokenized_corpus = [tokenisasi(doc_dict[doc_id]) for doc_id in doc_dict]

query = "sistem informasi statistik"
tokenized_query = tokenisasi(query)

bm25 = BM25Okapi(tokenized_corpus)

doc_scores = bm25.get_scores(tokenized_query)
print(doc_scores)

# menciptakan array query yang akan dihasilkan skor BM25
queries = [
    "sistem informasi statistik",
    "sistem",
    "informasi",
    "statistik"
]

# Inisialisasi BM25
bm25 = BM25Okapi(tokenized_corpus)

for query in queries:
    tokenized_query = tokenisasi(query)  # Tokenisasi query
    doc_scores = bm25.get_scores(tokenized_query)  # Hitung BM25 scores untuk query
    print(f"BM25 Scores untuk query '{query}':\n {doc_scores}" "\n")

from collections import OrderedDict
def exact_top_k(doc_dict, rank_score, k):
    relevance_scores = {}
    i = 0
    for doc_id in doc_dict.keys():
        relevance_scores[doc_id] = rank_score[i]
        i = i + 1

    sorted_value = OrderedDict(sorted(relevance_scores.items(), key = lambda x: x[1], reverse = True))
    top_k = {j: sorted_value[j] for j in list(sorted_value)[:k]}
    return top_k

k=3
print("\n",exact_top_k(doc_dict,doc_scores,k))
