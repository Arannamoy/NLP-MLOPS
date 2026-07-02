import matplotlib.pyplot as plt


year=[2021,2022,2023,2024,2025]
price_max=[60000,45000,16000,40000,120000]
price_min=[45000,34100,12000,29500,100000]

plt.plot(year,price_max,color="red",linestyle="--",marker="*",label="Max Price")
plt.plot(year,price_min,color="blue",linestyle="dashdot",marker="^",label="Min Price")

plt.title("Price Comparison")
plt.xlabel("Year")
plt.ylabel("Price")

plt.legend()

plt.savefig("./3_Matplotlib_Seaborn/Images/5_line_color.png")