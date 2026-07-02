import pandas as pd;

df=pd.read_json("data.json")

df1=df.fillna(0)

print(df1)

df2=df["Full Stack"].fillna("Ethvm")
df["Full Stack"]=df2
print(df)


df3=df["Age"].fillna(method="ffill") # fill from previous value

print(df3)


df4=df["Age"].fillna(method="bfill") # fil from next value
print(df["Age"].ffill())