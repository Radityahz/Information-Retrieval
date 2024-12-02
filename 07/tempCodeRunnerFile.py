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
