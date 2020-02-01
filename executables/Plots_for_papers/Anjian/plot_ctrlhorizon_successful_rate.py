import matplotlib.pyplot as plt
import numpy as np

"""
Plot of success rate for a single NN with different control horizons
"""

# testing_result = [47.74, 52.76, 61.81, 63.82, 50.75]
# baseline_result = [44.72, 45.73, 51.25, 52.26, 46.73]

testing_result = [48, 53, 62, 64, 51]
baseline_result = [45, 46, 51, 52, 47]

# labels = ['1.5', '1', '0.5', '0.25', '0.15']
labels = ['0.67', '1', '2', '4', '6.67']
# x = np.arange(len(labels))
x_coordinate = np.asarray([1, 2, 3, 4, 5])


fig, ax = plt.subplots()
ax.set_ylabel('Success rate (%)', fontsize=12)
ax.set_xlabel('Replanning frequency (Hz)', fontsize=12)
ax.set_title('Success rate with different replanning frequencies', fontsize=12)
ax.set_xticks(x_coordinate)
ax.set_xticklabels(labels)
ax.set_yticks(np.arange(45, 65, 5))

plt.plot(x_coordinate, testing_result, 'o-', label='WayPtNav-ReachabilityCost')
plt.plot(x_coordinate, baseline_result, 's-', label='WayPtNav-HeuristicsCost')



# for i, value in enumerate(testing_result):
#     x = x_coordinate[i]
#     y = testing_result[i]
#     if i == 0:
#         scatter = ax.scatter(x, y, marker='x', color='red', label='Ours')
#     else:
#         scatter = ax.scatter(x, y, marker='x', color='red')
#     ax.text(x + 0.05, y + 0.05, value, fontsize=9)
#
#
# for i, value in enumerate(baseline_result):
#     x = x_coordinate[i]
#     y = baseline_result[i]
#     if i == 0:
#         ax.scatter(x, y, marker='o', color='blue', label='Baseline')
#     else:
#         ax.scatter(x, y, marker='o', color='blue')
#     ax.text(x + 0.05, y + 0.05, value, fontsize=9)



ax.legend(loc='lower right')

ax.set_aspect(aspect=0.2)

plt.show()

plot_path = '/home/anjianl/Desktop/project/WayPtNav_paper/plots/ctrlhorizon_success_rate.png'
fig.savefig(plot_path)
