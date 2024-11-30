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

# Fungsi AND untuk melakukan intersect pada dua posting list
def intersect_two_lists(posting1, posting2):
    p1 = 0
    p2 = 0
    result = list()
    while p1 < len(posting1) and p2 < len(posting2):
        if posting1[p1] == posting2[p2]:
            result.append(posting1[p1])
            p1 += 1
            p2 += 1
        elif posting1[p1] < posting2[p2]:
            p1 += 1
        else:
            p2 += 1
    return result

#Optimasi query untuk list postings
def AND_optimized(postings):
    # Mengurutkan posting dari panjangnya list
    postings.sort(key=len)

    # Mulai dari list yang paling sedikit
    result = postings[0]

    for i in range(1, len(postings)):
        result = intersect_two_lists(result, postings[i])

        if not result:
            break

    return result

# Mendapatkan daftar dokumen 
posting_vaksin = list(inverted_index.get('vaksin', {}).keys())
posting_corona = list(inverted_index.get('corona', {}).keys())
posting_pfizer = list(inverted_index.get('pfizer', {}).keys())

# OR pada posting list
result = AND_optimized([posting_vaksin, posting_corona, posting_pfizer])

# Output hasil pencarian
if result:
    print(f"Query: vaksin AND corona AND pfizer")
    print(f"Ditemukan di dokumen:")
    for doc_id in result:
        print(f"{doc_names[doc_id]}")
else:
    print(f"vaksin AND corona AND pfizer tidak ditemukan di dokumen manapun.")