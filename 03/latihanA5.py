from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

#create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

doc1_term = ["pengembangan", "sistem", "informasi", "penjadwalan"]
doc2_term = ["pengembangan", "model", "analisis", "sentimen", "berita"]
doc3_term = ["analisis", "sistem", "input", "output"]

corpus_term = [doc1_term, doc2_term, doc3_term]
NUM_OF_DOCS = len(corpus_term)

#Inverted Index
inverted_index = {}

for i in range(len(corpus_term)):
    for item in corpus_term[i]:
        item = stemmer.stem(item)
        if item not in inverted_index:
            inverted_index[item] = []
        if (i+1) not in inverted_index[item]:
            inverted_index[item].append(i+1)
print("Inverted index: ", inverted_index)

#Fungsi NOT
def NOT(posting):
    result = []
    posting_set = set(posting)  # Convert to set for faster lookup
    for doc_id in range(1, NUM_OF_DOCS + 1):  # Loop through valid document IDs
        if doc_id not in posting_set:
            result.append(doc_id)
    return result

result = NOT(inverted_index['siste'])
print("Dokumen id yang tidak memiliki term sistem: ", result)