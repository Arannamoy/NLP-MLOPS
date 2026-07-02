import pandas as pd
import matplotlib.pyplot as plt




with plt.xkcd():
    data=pd.read_csv("8_Assignment_Data.csv")

    year=data["Year"]
    minPrice=data["MinPrice"]
    maxPrice=data["MaxPrice"]

    plt.plot(year,minPrice,color="green",marker="^",label="Minimum Price")
    plt.plot(year,maxPrice,color="red",marker="*",label="Maximum Price")

    plt.title("Price Comparison 2010-2025")

    plt.grid(True)

    plt.tight_layout()

    plt.legend()
    plt.savefig("./3_Matplotlib_Seaborn/Images/8_Assignment_Answer.png")