import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt



tips=sns.load_dataset("tips")

df=pd.DataFrame(tips)

print(df.info())

sns.lineplot(x="total_bill",y="tip",data=tips)
plt.title("Tips of a waiter in a restaurant.")
plt.savefig("./3_Matplotlib_Seaborn/Images/27_sns_load_dataste.png")