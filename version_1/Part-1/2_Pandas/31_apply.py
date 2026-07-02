import pandas as pd


df=pd.read_json("data.json")

df1=df.copy()

df1["Is Core Developer?"]=df1["Full Stack"].apply(lambda x: "Yes" if x=="MERN+Django+SpringBoot" else "No")

print(df1)


