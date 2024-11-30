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

#Casefolding dan Tokenisasi
for filename in files:
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

            # Mencetak hasil stemming
            print(f"Hasil stemming untuk {filename}:\n")
            print(stemmed_tokens)
            print("\n" + "="*50 + "\n")


   