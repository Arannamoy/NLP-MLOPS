import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



# with plt.xkcd():
data=pd.read_csv("9_Data.csv")

year=data["Year"]
minPrice=data["MinPrice"]
maxPrice=data["MaxPrice"]
x=np.arange(len(year))

width=0.25

plt.bar(x-width,minPrice,width=width,label="Minimum Price")
plt.bar(x,maxPrice,width=width,label="Maximum Price")

plt.title("Price Comparison 2010-2025")

# plt.grid(True)

plt.xlabel("Year")
plt.ylabel("price")

plt.xticks(x,year)
plt.tight_layout()

# plt.legend()
plt.savefig("./3_Matplotlib_Seaborn/Images/9_barchart.png")