import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("./6_Scikit_Learn/Data/housing.csv")
df=pd.DataFrame(data).copy()

df.hist(bins=50,figsize=(12,8))

plt.savefig("./6_Scikit_Learn/Data/histogram.png")

print(df['ocean_proximity'].value_counts())
print(df.describe())

