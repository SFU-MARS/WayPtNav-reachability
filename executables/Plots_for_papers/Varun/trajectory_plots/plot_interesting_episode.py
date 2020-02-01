import os
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pickle

linewidth = 3.5

area = 1

if area == 6:
    end_to_end_dir = './area6/end_to_end/session_2019-01-30_14-04-20/rgb_resnet50_nn_control_simulator/trajectories'
    waypoint_dir = './area6/waypoint/session_2019-01-30_13-57-31/rgb_resnet50_nn_waypoint_simulator/trajectories'

    interesting_goals = [40, 53, 56, 82]
    plotting_steps = {40: 6, 53: 6.5, 56: 5, 82: 6}
    quiver_freq = {40:30, 53: 30, 56: 30, 82: 30}
    goal_locations = {40: [13.4, 49.6], 53: [16.5, 34.85], 56: [11.5, 7.00], 82: [7.35, 12.65]}
    # interesting_goals = [40]
    # plotting_steps = {40: 6}
    # goal_locations = {40: [13.4, 49.6]}
    # planner_data_idx = {40: 11}
    # quiver_freq = {40: 10}
elif area == 1:
    # end_to_end_dir = './area1/end_to_end/session_2019-01-30_13-55-54/rgb_resnet50_nn_control_simulator/trajectories'
    # waypoint_dir = './area1/waypoint/session_2019-01-30_13-47-21/rgb_resnet50_nn_waypoint_simulator/trajectories'
    # interesting_goals = [31, 38, 73]
    # plotting_steps = {31: 5.5, 38: 6.0, 73:5}
    # quiver_freq = {31: 25, 38: 25, 73:25}
    # goal_locations = {31: [12.5, 24.7], 38: [13.15, 22.95], 73: [12.70, 25.15]}

    end_to_end_dir = '../bar_chart/end_to_end/session_2019-01-30_22-15-07/rgb_resnet50_nn_control_simulator/trajectories'
    waypoint_dir = '../bar_chart/waypoint/session_2019-01-30_22-25-51/rgb_resnet50_nn_waypoint_simulator/trajectories'
    interesting_goals = [73]
    plotting_steps = {73: 5}
    quiver_freq = {73: 25}
    goal_locations = {73: [12.70, 25.15]}
else:
    assert False


def plot_top_view(filename, mode, file_number):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    
    # Plot Map
    ax.imshow(data['occupancy_grid'], cmap='gray_r', extent=data['map_bounds_extent'],
              vmax=1.5, vmin=-.5, origin='lower')

    pos_k2 = data['vehicle_trajectory']['position_nk2'][0]
    
    # Plot Trajectory
    ax.plot(pos_k2[:, 0], pos_k2[:, 1], 'r-', linewidth=linewidth, zorder=1)
    
    # Plot Waypoints Or Quiver
    if 'waypoint_config' in data['vehicle_data'].keys():
        system_pos_n2 = data['vehicle_data']['system_config']['position_nk2'][:, 0]
        system_heading_n1 = data['vehicle_data']['system_config']['heading_nk1'][:, 0]
        ax.quiver(system_pos_n2[:, 0][::2], system_pos_n2[:, 1][::2], np.cos(system_heading_n1[::2]), np.sin(system_heading_n1[::2]), zorder=5)
    else:
        heading_k1 = data['vehicle_trajectory']['heading_nk1'][0]
        ax.quiver(pos_k2[:, 0][::quiver_freq[file_number]], pos_k2[:, 1][::quiver_freq[file_number]],
            np.cos(heading_k1[::quiver_freq[file_number]]), np.sin(heading_k1[::quiver_freq[file_number]]), zorder=5)

    # Plot the initial state
    start_2 = data['vehicle_trajectory']['position_nk2'][0, 0]
    ax.plot(start_2[0], start_2[1], 'bo', markersize=14)

    # Plot the Goal
    ax.plot(goal_locations[file_number][0], goal_locations[file_number][1], 'k*')
    center = goal_locations[file_number]
    radius = .3
    c = plt.Circle(center, radius, color='g')
    ax.add_artist(c)

    # Set the extent of the plot
    delta = plotting_steps[file_number]
    max_x = max(start_2[0], goal_locations[file_number][0])
    min_x = min(start_2[0], goal_locations[file_number][0])
    max_y = max(start_2[1], goal_locations[file_number][1])
    min_y = min(start_2[1], goal_locations[file_number][1])
    # print(max_x, min_x, max_y, min_y)
    
    max_size = 11
    delta_x = max_size - (max_x - min_x)
    delta_y = max_size - (max_y - min_y)
    ax.set_xlim(min_x-0.5*delta_x, max_x+0.5*delta_x)
    ax.set_ylim(min_y-0.5*delta_y, max_y+0.5*delta_y)
    # ax.set_xlim(start_2[0]-delta, start_2[0]+delta)
    # ax.set_ylim(start_2[1]-delta, start_2[1]+delta)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the plot
    fig.savefig('./plots/area{:d}_traj_{:d}_{:s}.pdf'.format(area, file_number, mode), bbox_inches='tight', dpi=300)
    # fig.savefig('./plots/area{:d}_traj_{:d}_{:s}.png'.format(area, file_number, mode), bbox_inches='tight')

def plot_interesting_episodes(mode):
    if mode == 'end_to_end':
        directory_name = end_to_end_dir
    elif mode == 'waypoint':
        directory_name = waypoint_dir
    else:
        assert False

    assert(os.path.exists(directory_name))
    data_files = os.listdir(directory_name)
    data_files = list(filter(lambda x: 'metadata' not in x, data_files))

    for data_file in data_files:
        file_number = int(data_file.split('.pkl')[0][5:])
        if file_number in interesting_goals:
            delta = plotting_steps[file_number]
            plot_top_view(os.path.join(directory_name, data_file), mode, file_number)

if __name__ == '__main__':
    matplotlib.style.use('ggplot')
    plot_interesting_episodes(mode='waypoint')
    plot_interesting_episodes(mode='end_to_end')
