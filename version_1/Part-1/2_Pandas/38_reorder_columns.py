import pandas as pd


df=pd.read_json("data.json")

df1=df.copy()

# df1.columns=df[["Name","Full Stack","Age","Profession",.......]]
print(df1)