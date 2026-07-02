import matplotlib.pyplot as plt


year=[2017,2018,2019,2020,2021,2022,2023,2024,2025]
total_languages=[2,0,3,0,0,0,4,6,6]

plt.plot(year,total_languages)

plt.show()

plt.savefig("./3_Matplotlib_Seaborn/Images/1_Intro.png")