import matplotlib
import matplotlib.pyplot as plt
import numpy as np

"""
Plot of success rate for episodes for each difficulty level, measured by the minimum distance to the obstacles in expert
trajectories
"""

# labels = ['<0.15', '0.15-0.2', '0.2-0.25', '0.25-0.3', '0.3-0.4', '>0.4']
labels = ['Hard (<0.2m)\n242 tasks', 'Medium (0.2m-0.3m)\n155 tasks', 'Easy (>0.3m)\n102 tasks']

# ours_means = [69, 48, 60, 82, 87, 83]
# baseline_means = [33, 23, 53, 79, 87, 100]

ours_means = [52.06, 69.68, 86.27]
baseline_means = [25.20, 64.52, 89.21]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, ours_means, width, label='WayPtNav-ReachabilityCost')
rects2 = ax.bar(x + width/2, baseline_means, width, label='WayPtNav-HeuristicsCost')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Success rate (%)', fontsize=13)
ax.set_xlabel('Difficult levels', fontsize=13)
ax.set_title('Success rate at different difficulty levels', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_yticks(np.arange(0, 100, 25))
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


# autolabel(rects1)
# autolabel(rects2)

fig.tight_layout()

# ax.set_aspect(aspect=0.1)

plt.show()

plot_path = '/home/anjianl/Desktop/project/WayPtNav_paper/plots/obsdist_success_rate.png'
fig.savefig(plot_path)
