import matplotlib.pyplot as plt


study_hours = [1, 2, 3, 4, 5, 6, 7, 8, 9]
exam_scores = [40, 35, 20, 55, 60, 65, 75, 85, 90]

plt.scatter(study_hours,exam_scores,c=exam_scores,cmap="viridis")

plt.savefig("./3_Matplotlib_Seaborn/Images/21_cmap_scatter.png")