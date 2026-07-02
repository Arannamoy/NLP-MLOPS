import matplotlib.pyplot as plt



extensions=["py","ipynb","json","csv","xlsx"]
ratio=["80","10","2.5","5","2.5"]

plt.pie(ratio,labels=extensions,explode=[0.4,0,0,0,0.1],autopct="%1.1f")

plt.savefig("./3_Matplotlib_Seaborn/Images/17_pie_autopct.png")