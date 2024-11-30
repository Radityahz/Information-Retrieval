import re

def paragraph_parsing(dokumen):
    list_paragraf = dokumen.split('\n')
    return list_paragraf

def sentence_parsing(paragraf):
    list_kalimat = re.split(r'(?<=[.!?]) +', paragraf)
    return list_kalimat

dokumen = """Mobilitas warga bakal diperketat melalui penerapan PPKM lebel 3 se-Indonesia di masa libur Natal dan tahun baru (Nataru). Rencana kebijakan itu dikritik oleh Epidemilog dari Griffith University Dicky Budiman.
Dicky menyebut pembatasan mobilitas memang akan memiliki dampak dalam mencegah penularan COVID-19. Tapi, kata dia, dampaknya signifikan atau tidak akan bergantung pada konsistensi yang mendasar yakni"""

#Memisahkan dokumen menjadi list paragraf
list_paragraf = paragraph_parsing(dokumen)

#Mencetak hasil parsing_paragraf
print("List paragraf:")
for i, paragraf in enumerate(list_paragraf):
    print(f"p{i+1}: {paragraf}\n")

#Memisahkan paragraf menjadi kalimat
list_kalimat= sentence_parsing(list_paragraf[0])

#Mencetak hasil sentence_parsing
print("List kalimat pada paragraf 1:")
for j, kalimat in enumerate(list_kalimat):
    print(f"s{j+1}: {kalimat.strip()}") 
