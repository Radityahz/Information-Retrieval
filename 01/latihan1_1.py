#Import Module
import os

#Folder Path
path = r"C:\\Users\\R1N0C\\IR\\berita"

#List all files in a directory using os.listdir
for file in os.listdir(path):
    if os.path.isfile(os.path.join(path, file)):
        print(file)

def read_text_file(file_path) :
    with open(file_path, 'r') as f:
        print(f.read())

#iterate through all file
for file in os.listdir(path):
    #check whether file is in text format or not
    if file.endswith(".txt"):
        file_path = f"{path}\{file}"

        #call read text file function
        read_text_file(file_path)
