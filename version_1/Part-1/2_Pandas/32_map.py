import pandas as pd


df=pd.read_json("data.json")


o_map={"MERN+Django+SpringBoot":"Core Developer"}

df1=df.copy()

print(df1["Full Stack"].map(o_map))