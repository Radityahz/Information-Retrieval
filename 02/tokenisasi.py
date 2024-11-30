#fungsi tokenisasi 
def tokenisasi(text):
    tokens = text.split(" ")
    return tokens

doc1 = "pengembangan sistem informasi penjadwalan"
doc2 = "pengembangan model analisis sentimen berita"
doc3 = "pengembangan model analisis sentimen berita"
corpus = [doc1, doc2, doc3]

for d in corpus:
    token_kata = tokenisasi(d)
    print(token_kata)

from spacy.lang.id import Indonesian
import spacy
nlp = Indonesian() # use directly
nlp = spacy.blank('id') #blank instance'
for d in corpus:
    spacy_id = nlp(d)
    token_kata = [token.text for token in spacy_id]
    print(token_kata)
