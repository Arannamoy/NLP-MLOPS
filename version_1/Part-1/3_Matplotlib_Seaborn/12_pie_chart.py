import matplotlib.pyplot as plt


extensions=["py","ipynb","json","csv","xlsx"]
ratio=["80","10","2.5","5","2.5"]

plt.pie(ratio,labels=extensions)

plt.savefig("./3_Matplotlib_Seaborn/Images/12_Pie_chart.png")