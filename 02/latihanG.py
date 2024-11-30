from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
#create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()
#stemming process
text = "Wilayah Kamu Sudah 'Bebas' COVID-19? Cek 34 Kab/Kota Zona Hijau Terbaru"
output = stemmer.stem(text)
print(output)