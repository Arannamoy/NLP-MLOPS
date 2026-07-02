import requests

data=requests.get("https://quotes.toscrape.com/")

with open (f"./4_Data_Collection/data/page1.html","a") as fl:
    fl.write(data.text)