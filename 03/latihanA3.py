from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

#create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

doc1_term = ["pengembangan", "sistem", "informasi", "penjadwalan"]
doc2_term = ["pengembangan", "model", "analisis", "sentimen", "berita"]
doc3_term = ["analisis", "sistem", "input", "output"]

corpus_term = [doc1_term, doc2_term, doc3_term]

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

#Boolean Retrieval
def AND(posting1, posting2, posting3):
    p1 = 0
    p2 = 0
    p3 = 0
    result = []
    while p1 < len(posting1) and p2 < len(posting2) and p3 < len(posting3):
        if posting1[p1] == posting2[p2] == posting3[p3]:
            result.append(posting1[p1])
            p1 += 1
            p2 += 1
            p3 += 1
        elif posting1[p1] < posting2[p2]:
            p1 += 1
        elif posting2[p2] < posting3[p3]:
            p2 += 1
        else:
            p3 += 1
    return result

result = AND(inverted_index['sistem'], inverted_index['informasi'], inverted_index['jadwal'])
print("Dokumen id untuk sistem, informasi, dan jadwal: ", result)
