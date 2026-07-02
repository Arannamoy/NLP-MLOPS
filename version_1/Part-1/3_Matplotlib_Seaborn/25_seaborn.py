# Matplotlib is powerful but very low-level.
# Seaborn adds high-level features like automatic styling, themes, color palettes, and dataframe integration.
# Seaborn comes with built in Datasets
# Makes complex plots (like boxplots, violin plots, heatmaps, pairplots) very easy.

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

x=np.array(np.arange(0,100,30))
y=np.sin(x)

sns.lineplot(x=x,y=y)
plt.savefig("./3_Matplotlib_Seaborn/Images/25_seaborn.png")





