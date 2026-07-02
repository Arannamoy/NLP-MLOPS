import matplotlib.pyplot as plt

study_hours = [1, 2, 3, 4, 5, 6, 7, 8, 9]
exam_scores = [40, 35, 20, 55, 60, 65, 75, 85, 90]


plt.scatter(study_hours, exam_scores)

# Add labels
for i in range(len(study_hours)):
    plt.annotate(f'Student {i+1}', (study_hours[i], exam_scores[i]))

plt.title('Scatter Plot with Annotations')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.grid(True)
plt.savefig("./3_Matplotlib_Seaborn/Images/23_plt_scatter_annotations.png")