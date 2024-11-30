#Import modul
import os
import re
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
path = r"C:\\Users\\R1N0C\\IR\\P2\\berita"

# Mengambil semua file dalam folder
files = os.listdir(path)

# Inverted index
inverted_index = {}

#Casefolding dan Tokenisasi
for doc_id, filename in enumerate(files, start=1):
    # Memastikan hanya memproses file teks
    if filename.endswith('.txt'):
        file_path = os.path.join(path, filename)
        
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

            #Inverted Index
            for pos, term in enumerate(stemmed_tokens):  # `pos` adalah lokasi token di dokumen
                if term not in inverted_index:
                    inverted_index[term] = {}
                if doc_id not in inverted_index[term]:
                    inverted_index[term][doc_id] = []
                inverted_index[term][doc_id].append(pos)

def boolean_retrieval(term):
    # Lakukan stemming pada term yang dimasukkan
    stemmed_term = stemmer.stem(term.lower())
    
    # Cek apakah term ada dalam inverted index
    if stemmed_term in inverted_index:
        return sorted(list(inverted_index[stemmed_term]))  # Kembalikan dokumen yang mengandung term
    else:
        return []

term = "corona"
result = boolean_retrieval(term)

# Output hasil pencarian
if result:
    print(f"Term '{term}' ditemukan di dokumen: {result}")
else:
    print(f"Term '{term}' tidak ditemukan di dokumen manapun.")

term = "covid"
result = boolean_retrieval(term)

# Output hasil pencarian
if result:
    print(f"Term '{term}' ditemukan di dokumen: {result}")
else:
    print(f"Term '{term}' tidak ditemukan di dokumen manapun.")

term = "vaksin"
result = boolean_retrieval(term)

# Output hasil pencarian
if result:
    print(f"Term '{term}' ditemukan di dokumen: {result}")
else:
    print(f"Term '{term}' tidak ditemukan di dokumen manapun.")
