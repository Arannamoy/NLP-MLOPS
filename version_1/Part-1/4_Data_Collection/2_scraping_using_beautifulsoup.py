from bs4 import BeautifulSoup
with open ("./4_Data_Collection/data/page1.html") as fl:
    content=fl.read()

soup=BeautifulSoup(content,"html.parser")

h3s=soup.find_all("h3")

print(h3s)