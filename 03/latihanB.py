from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

doc1_term = ["pengembangan", "sistem", "informasi", "penjadwalan"]
doc2_term = ["pengembangan", "model", "analisis", "sentimen", "berita"]
doc3_term = ["analisis", "sistem", "input", "output"]

corpus_term = [doc1_term, doc2_term, doc3_term]

# Inverted Index
inverted_index = {}

for i in range(len(corpus_term)):
    for item in corpus_term[i]:
        item = stemmer.stem(item)  # Perform stemming
        if item not in inverted_index:
            inverted_index[item] = []
        # Add document ID if it's not already in the list
        if (i + 1) not in inverted_index[item]:
            inverted_index[item].append(i + 1)

# Print the inverted index
print("Inverted Index: ", inverted_index)

# Boolean AND function
def AND(posting1, posting2):
    p1 = 0
    p2 = 0
    result = []
    while p1 < len(posting1) and p2 < len(posting2):
        if posting1[p1] == posting2[p2]:
            result.append(posting1[p1])
            p1 += 1
            p2 += 1
        elif posting1[p1] > posting2[p2]:
            p2 += 1
        else:
            p1 += 1
    return result

# Perform AND operation on "sistem" and "analisis"
result = AND(inverted_index['sistem'], inverted_index['analisis'])

# Print the result of the Boolean retrieval
print("Hasil Boolean:", result)


