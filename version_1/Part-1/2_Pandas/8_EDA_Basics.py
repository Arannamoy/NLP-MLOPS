# EDA --> Exploratory Data Analysis
# df.head(),df.tail(),df.describe(),df.info(),df.columns,df.shape


import pandas as pd

df=pd.read_csv("mobile-code.csv")

print("**********\n\tHead:\n",df.head())
print("**********\n\tTail:\n",df.tail())
print("**********\n\tInfo:\n",df.info())
print("**********\n\tDescribe:\n",df.describe())
print("**********\n\tColumns:\n",df.columns)
print("**********\n\tShape:\n",df.shape)
