import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

files = ["weighted_average", "adj", "common_users", "varians"]

for file in files:
    df = pd.read_csv(file+'.csv')

    averages = df.mean()

    categories = ['PasP', 'PasN', 'NasN', 'NasP']

    average_values = [averages['PasP'], averages['PasN'], averages['NasN'], averages['NasP']]

    confusion_matrix_data = pd.DataFrame({'Predicted Positive': [average_values[0], average_values[3]],
                                        'Predicted Negative': [average_values[1], average_values[2]]},
                                        index=['Actual Positive', 'Actual Negative'])

    sns.heatmap(confusion_matrix_data, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title('Confusion Matrix of Averages for ' + file)
    plt.show()




files = ["weighted_average", "adj", "common_users", "varians"]

mae_avg = []
mar_avg = []
map_avg = []

for file in files:
    df = pd.read_csv(file+'.csv')

    mae_avg.append(df['MAE'].mean())
    mar_avg.append(df['MAR'].mean())
    map_avg.append(df['MAP'].mean())


plt.figure(figsize=(10, 6))

bar_width = 0.25

r1 = np.arange(len(files))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

bars1 = plt.bar(r1, mae_avg, color='b', width=bar_width, edgecolor='grey', label='MAE')
bars2 = plt.bar(r2, mar_avg, color='g', width=bar_width, edgecolor='grey', label='MAR')
bars3 = plt.bar(r3, map_avg, color='r', width=bar_width, edgecolor='grey', label='MAP')

def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height, '%.2f' % height, ha='center', va='bottom')

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

plt.xlabel('File', fontweight='bold')
plt.ylabel('Average', fontweight='bold')
plt.title('Average Results of MAE, MAR, and MAP for Each File', fontweight='bold')

plt.xticks([r + bar_width for r in range(len(files))], files)

plt.legend()

plt.tight_layout()
plt.show()
