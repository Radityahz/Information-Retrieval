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

#preprosesing
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

# Mencetak hasil inverted index
print("Inverted Index (Term -> Daftar Lokasi (posting lists)):\n")
for term, posting in inverted_index.items():
    print(f"{term} -> {sorted(list(posting))}")
