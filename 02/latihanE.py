stopwords = ["yang", "dari", "sudah", "dan"]
text = "Wilayah Kamu Sudah 'Bebabs' COVID-19? Cek 34 Kab/Kota Zona Hijau Terbaur"
tokens = ["wilayah", "kamu", "sudah", "bebas", "covid-19", "?", "cek", "34", "kab", 
"/", "kota", "zona", "hijau", "terbaru"]
tokens_nostopword = [w for w in tokens if not w in stopwords]
print(tokens_nostopword)

from spacy.lang.id import Indonesian
import spacy
nlp = Indonesian() #use directly
nlp = spacy.blank('id') #blank instance'
stopwords = nlp.Defaults.stop_words
tokens_nostopword = [w for w in tokens if not w in stopwords]
print(tokens_nostopword)
