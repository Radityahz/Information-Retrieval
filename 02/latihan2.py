doc1 = "pengembangan sistem informasi penjadwalan"
doc1_term = ["pengembangan", "sistem", "informasi", "penjadwalan"]
doc2 = "pengembangan model analisis sentimen berita"
doc2_term = ["pengembangan", "model", "analisis", "sentimen", "berita"]
doc3 = "pengembangan model analisis sentimen berita"
doc3_term = ["analisis", "sistem", "input", "output"]

corpus = [doc1, doc2, doc3]
corpus_term = [doc1_term, doc2_term, doc3_term]

#Menggabungkan term ke dalam vocabulary
vocabulary = {}

for d in corpus_term:
    for term in d:
        if term not in vocabulary:
            vocabulary[term] = 1
        else:
            vocabulary[term] = vocabulary[term]+1
print(vocabulary)