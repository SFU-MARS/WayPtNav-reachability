import os
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import skfmm

linewidth = 3.5
file_number = 0
plotting_steps = {0: 5.5}
quiver_freq = {0: 15}
goal_locations = {0: [4.0, 0.0]}
traj_origin = [1.5, 0.]
traj_angle = -0.5*np.pi

traj_file = '/home/somilb/Documents/Projects/visual_mpc/tmp/paper_visualizations_v2/experiment_results/' \
            'experiment3/top_view/traj.png'
map_file = '/home/ext_drive/somilb/data/experiments/CORL_2019/SLAM_MAPS/sdh7_sunlight_later_bigger/data.pkl'
exp_data1 = '/home/ext_drive/somilb/data/experiments/CORL_2019/final/Sunny SDH7 Experiment/WayPtNav/' \
            'session_2019-07-03_18-02-29/rgb_resnet50_nn_waypoint_simulator/trajectories'

def shift_coordinates(pos_k2):
    # rotate the entire trajectory by 90 degrees
    pos_new_k2 = pos_k2 * 1.
    pos_new_k2[:, 0] = pos_k2[:, 0]
    pos_new_k2[:, 1] = pos_k2[:, 1]
    
    # Shift the origin
    pos_new_k2[:, 0] = pos_new_k2[:, 0] + traj_origin[0]
    pos_new_k2[:, 1] = pos_new_k2[:, 1] + traj_origin[1]
    
    return pos_new_k2

def rotate_angles(theta_k1):
    # rotate the entire trajectory by 90 degrees
    return theta_k1

def plot_top_view(goal_locations):
    # Get the map data
    data = pickle.load(open(map_file, "rb"), encoding='latin1')
    map_data = data['map_data']
    map_extent = data['map_extent']
    # map_extent = np.array([map_extent[0], map_extent[2], map_extent[1], map_extent[3]])
    
    # Augment the map to include robot base
    signed_dist_to_obs = skfmm.distance(-1. * map_data + 1e-2, dx=0.05)
    occupancy_map_mn = 0.5 * (1 + np.sign(0.05 - signed_dist_to_obs))
    occupancy_map_mn = np.clip(occupancy_map_mn, a_min=0.0, a_max=1.0)
    occupancy_map_mn = -10*np.clip(map_data, a_min=-1.0, a_max=0.0) + 100*occupancy_map_mn
    
    # Get the trajectory data
    data1 = pickle.load(open(os.path.join(exp_data1, 'traj_0.pkl'), "rb"), encoding='latin1')

    # Start plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    # Plot Map
    ax.imshow(occupancy_map_mn, cmap='gray_r', extent=map_extent, vmax=30.0, vmin=0.0, origin='lower')

    # Plot Trajectory
    pos_k2 = data1['vehicle_trajectory']['position_nk2'][0]
    pos_k2 = shift_coordinates(pos_k2)
    ax.plot(pos_k2[:, 0], pos_k2[:, 1], 'r-', linewidth=linewidth, zorder=1)

    # # Plot Waypoints Or Quiver
    # if 'waypoint_config' in data1['vehicle_data'].keys():
    #     system_pos_n2 = data1['vehicle_data']['system_config']['position_nk2'][:, 0]
    #     system_pos_n2 = shift_coordinates(system_pos_n2)
    #     system_heading_n1 = data1['vehicle_data']['system_config']['heading_nk1'][:, 0]
    #     system_heading_n1 = rotate_angles(system_heading_n1)
    #     ax.quiver(system_pos_n2[:, 0][::2], system_pos_n2[:, 1][::2], np.cos(system_heading_n1[::2]),
    #               np.sin(system_heading_n1[::2]))
    # else:
    #     heading_k1 = data1['vehicle_trajectory']['heading_nk1'][0]
    #     heading_k1 = rotate_angles(heading_k1)
    #     ax.quiver(pos_k2[:, 0][::quiver_freq[file_number]], pos_k2[:, 1][::quiver_freq[file_number]],
    #               np.cos(heading_k1[::quiver_freq[file_number]]), np.sin(heading_k1[::quiver_freq[file_number]]))
    #
    # # Plot Waypoints Or Quiver
    # if 'waypoint_config' in data2['vehicle_data'].keys():
    #     system_pos_n2 = data2['vehicle_data']['system_config']['position_nk2'][:, 0]
    #     system_pos_n2 = shift_coordinates(system_pos_n2)
    #     system_heading_n1 = data2['vehicle_data']['system_config']['heading_nk1'][:, 0]
    #     system_heading_n1 = rotate_angles(system_heading_n1)
    #     ax.quiver(system_pos_n2[:, 0][::2], system_pos_n2[:, 1][::2], np.cos(system_heading_n1[::2]),
    #               np.sin(system_heading_n1[::2]))
    # else:
    #     heading_k1 = data2['vehicle_trajectory']['heading_nk1'][0]
    #     heading_k1 = rotate_angles(heading_k1)
    #     ax.quiver(pos_k2[:, 0][::quiver_freq[file_number]], pos_k2[:, 1][::quiver_freq[file_number]],
    #               np.cos(heading_k1[::quiver_freq[file_number]]), np.sin(heading_k1[::quiver_freq[file_number]]))

    # Plot the initial state
    start_k2 = data1['vehicle_trajectory']['position_nk2'][0]
    start_2 = shift_coordinates(start_k2)[0]
    ax.plot(start_2[0], start_2[1], 'bo', markersize=14)

    # Plot the Goal
    goal_locations0 = np.array(goal_locations[0])[None, :]
    goal_locations0 = shift_coordinates(goal_locations0)[0]
    ax.plot(goal_locations0[0], goal_locations0[1], 'k*')
    center = goal_locations0
    radius = .3
    c = plt.Circle(center, radius, color='g')
    ax.add_artist(c)

    # Set the extent of the plot
    ax.grid(False)
    # ax.set_xlim(10, 22.5)
    # ax.set_ylim(42.5, 55)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Save the plot
    fig.savefig(traj_file, bbox_inches='tight')
    

if __name__ == '__main__':
    plot_top_view(goal_locations)
