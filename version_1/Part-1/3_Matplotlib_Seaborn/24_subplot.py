# multiple plot in a single figure

import matplotlib.pyplot as plt


x = [1, 2, 3, 4, 5]
y1 = [i * 2 for i in x]
y2 = [i ** 2 for i in x]


plt.subplot(1,2,1)
plt.plot(x,y1)


plt.subplot(1,2,2)
plt.plot(x,y2)

plt.savefig("./3_Matplotlib_Seaborn/Images/24_subplot.png")