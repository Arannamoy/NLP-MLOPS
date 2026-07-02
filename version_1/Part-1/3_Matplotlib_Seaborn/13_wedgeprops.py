import matplotlib.pyplot as plt


extensions=["py","ipynb","json","csv","xlsx"]
ratio=["80","10","2.5","5","2.5"]

plt.pie(ratio,labels=extensions,wedgeprops={"edgecolor":"black","linewidth":1,"linestyle":"--"})

plt.savefig("./3_Matplotlib_Seaborn/Images/13_Wedgeprops.png")