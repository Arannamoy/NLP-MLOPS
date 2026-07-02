import matplotlib.pyplot as plt

import random

data=[random.randint(1,100) for x in range(0,100)]

# plt.hist(data,bins=10,edgecolor="black")

plt.hist(data,bins=[10,20,30,40,50,60,70,80,90,100],edgecolor="black")
plt.axvline(15,color="red",linestyle="dotted",linewidth=4) # adline based 1st param

plt.xlabel("Marks")
plt.ylabel("Frequency")

plt.grid(True,linestyle="--",alpha=0.5)

plt.savefig("./3_Matplotlib_Seaborn/Images/19_plt_hist.png")