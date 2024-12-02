from bs4 import BeautifulSoup as Soup
import csv
import re
import requests

def downloader(link):
    req = requests.get(link)
    req.encoding = "utf8"
    return req.text

contents = downloader("https://jurnal.stis.ac.id/index.php/jurnalasks/")

#print(contents)

soup = Soup(contents, "lxml")
#print("\n", soup.prettify(), "\n")

#soup.title

#soup.find_all("a", attrs={"id": "article-532"})

urls = soup.find_all("a", attrs={"id": re.compile(r"(article)")})

#print()
"""for u in urls:
    content_u = downloader(u['href'])
    soup_u = Soup(content_u, "lxml")
    print(soup_u.find("h1", attrs={"class": "page_title"}).text)"""

with open('authors.csv', mode='w', newline='', encoding='utf8') as file:
    writer = csv.writer(file)
    writer.writerow(['Judul Jurnal', 'Author'])

    for u in urls:
        article_url = u['href']
        content_u = downloader(article_url)

        soup_u = Soup(content_u, "lxml")

        title = soup_u.find("h1", attrs={"class": "page_title"}).text.strip()

        authors_tag = soup_u.find("ul", attrs={"class": "authors"})
        authors = [author.text.strip() for author in authors_tag.find_all("span", class_="name")]
        authors_combined = ", ".join(authors)

        for author in authors:
            writer.writerow([title, author.strip()])

print("Judul Artikel dan Nama Author telah di simpan di 'authors.csv'.")