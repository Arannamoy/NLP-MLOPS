import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


flights=sns.load_dataset("flights")

pv=flights.pivot(index="month",columns="year",values="passengers")

sns.heatmap(pv,annot=True,fmt="d",cmap="YlGnBu")

plt.savefig("./3_Matplotlib_Seaborn/Images/28_sns_heatmap.png")

