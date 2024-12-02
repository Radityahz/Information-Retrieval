import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
import json

# Pastikan Anda mengunduh paket NLTK yang diperlukan
nltk.download('punkt')

# Path file CSV
file_path = r"E:\KULIAH\TINGKAT 3\IR\PRAKTIKUM\preprocessing\test.csv"

# Mencoba membaca file dengan beberapa opsi untuk menghindari ParserError
try:
    # Membaca file CSV dengan delimiter default (koma) atau sesuaikan dengan format
    df = pd.read_csv(file_path, delimiter=';', on_bad_lines='skip', engine='python')  # Abaikan baris bermasalah
    print(df.columns)  # Menampilkan nama kolom
    print(df.head())   # Menampilkan beberapa baris data
    print("File CSV berhasil dibaca.")
except Exception as e:
    print("Terjadi error saat membaca file CSV:", e)
    exit()

# Fungsi untuk tokenisasi dan menggabungkan isi kolom tertentu
def preprocess_row(row):
    # Pilih kolom 1, 2, dan 3 (gunakan indeks atau nama kolom)
    selected_columns = row[['r11a', 'r11b', 'r11c']]  # Pilih kolom yang sesuai
    # Gabungkan isi kolom menjadi satu string
    combined_text = ' '.join(selected_columns.astype(str))  # Konversi ke string
    # Tokenisasi
    tokens = word_tokenize(combined_text)
    return combined_text, tokens

# Pastikan dataframe tidak kosong sebelum diproses
if not df.empty:
    # Proses setiap baris dan tambahkan kolom baru
    df[['teks_gabungan', 'tokenisasi']] = df.apply(
        lambda row: pd.Series(preprocess_row(row)), axis=1
    )

    # Ubah format token menjadi string JSON agar bisa dibaca sebagai list saat di-load kembali
    df['tokenisasi'] = df['tokenisasi'].apply(lambda x: json.dumps(x))  # Mengubah list token menjadi string JSON

    # Path untuk file hasil
    output_path = r"E:\KULIAH\TINGKAT 3\IR\PRAKTIKUM\preprocessing\hasil.csv"

    # Menyimpan hasil ke file baru tanpa menambahkan indeks
    df.to_csv(output_path, index=False)
    print("Preprocessing selesai. Hasil disimpan di:", output_path)
else:
    print("DataFrame kosong. Tidak ada data untuk diproses.")