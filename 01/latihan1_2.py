text_1 = "Wilayah kamu Sudah 'Bebas' COVID-19? Cek 34 Kab/Kota Zona Hijau Terbaru"
text_2 = "Vaksin COVID-19 Bakal Rutin Setiap Tahun? Tergantung, Ini Penjelasannya"
text_3 = "RI Mulai Suntikan Booster di 2022, Masihkah Ampuh Lawan Varian Delta Cs?"
query = "COVID-19"
docs = [text_1, text_2, text_3]
for doc in docs: 
    if query in doc:
        print(doc)

import re
for doc in docs:
    if re.search(query, doc):
        print(doc)