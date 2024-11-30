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
NUM_OF_DOCS = len(files)

# Inverted index
inverted_index = {}
doc_names = {}  # Menyimpan nama dokumen berdasarkan ID

#Preprocessing dan Inverted Index
for doc_id, filename in enumerate(files, start=1):
    # Memastikan hanya memproses file teks
    if filename.endswith('.txt'):
        file_path = os.path.join(path, filename)
        doc_names[doc_id] = filename  # Menyimpan nama file berdasarkan ID
        
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

            # Inverted Index Construction
            for pos, term in enumerate(stemmed_tokens):  
                if term not in inverted_index:
                    inverted_index[term] = {}
                if doc_id not in inverted_index[term]:
                    inverted_index[term][doc_id] = []
                inverted_index[term][doc_id].append(pos)

#Fungsi NOT
def NOT(posting):
    result = []
    posting_set = set(posting)  # Convert to set for faster lookup
    for doc_id in range(1, NUM_OF_DOCS + 1):  # Loop through valid document IDs
        if doc_id not in posting_set:
            result.append(doc_id)
    return result

# Mendapatkan daftar dokumen 
posting_vaksin = list(inverted_index.get('vaksin', {}).keys())

# OR pada posting list
result = NOT(posting_vaksin)

# Output hasil pencarian
if result:
    print(f"Query: NOT vaksin")
    print(f"Ditemukan di dokumen:")
    for doc_id in result:
        print(f"{doc_names[doc_id]}")
else:
    print(f"Query NOT vaksin tidak ditemukan di dokumen manapun.")