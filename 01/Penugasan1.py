#Import Module
import os

#Folder Path
path = r"C:\\Users\\R1N0C\\IR\\berita"

# Initialize lists
docs = []
judul_docs = []

# Function to read text file
def read_text_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()

#iterate through all file
for file in os.listdir(path):
    #check whether file is in text format or not
    if file.endswith(".txt"):
        file_path = os.path.join(path, file)

        #call read text file function
        docs.append(read_text_file(file_path))

        #append the title of the document
        judul_docs.append(file)

query = "corona"

#display document titles that contains the querry
matching_titles =[]
for i, doc in enumerate(docs):
    if query.lower() in doc.lower():
        matching_titles.append(judul_docs[i])

#print the list of documents
if matching_titles:
    print(f"Berikut Hasil pencarian:")
    for title in matching_titles:
        print(f"- {title}")
else:
    print(f"Hasil pencarian tidak tersedia")