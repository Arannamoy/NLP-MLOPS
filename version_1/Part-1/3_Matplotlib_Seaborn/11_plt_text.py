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

plt.bar(x-width,minPrice,color="red",label="Minimum Price")
plt.bar(x,maxPrice,color="green",label="Maximum Price")

plt.title("Price Comparison 2010-2025")

# plt.grid(True)

for i in range(len(year)):
    plt.text(i,minPrice[i]+50,str(minPrice[i]),ha="center")
    plt.text(i,maxPrice[i]+50,str(maxPrice[i]),ha="center")

plt.xlabel("Year")
plt.ylabel("price")

plt.xticks(x,year)
plt.tight_layout()

# plt.legend()
plt.savefig("./3_Matplotlib_Seaborn/Images/11_plt_text.png")