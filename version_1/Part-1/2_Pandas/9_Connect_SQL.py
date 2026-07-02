import sqlite3
import pandas as pd

db=sqlite3.connect("name.sqlite")
df=pd.read_sql("SELECT * FROM PERSON",db)