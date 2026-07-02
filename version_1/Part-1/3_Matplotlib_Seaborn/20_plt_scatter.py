import matplotlib.pyplot as plt


study_hours = [1, 2, 3, 4, 5, 6, 7, 8, 9]
exam_scores = [40, 35, 20, 55, 60, 65, 75, 85, 90]

colors=["red" if score<40 else "green" for score in exam_scores]

plt.scatter(study_hours,exam_scores,color=colors)

plt.savefig("./3_Matplotlib_Seaborn/Images/20_plt_scatter.png")