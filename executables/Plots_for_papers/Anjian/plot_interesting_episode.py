import os
import matplotlib

matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pickle

import tensorflow as tf
import argparse
import importlib
import os

from utils import utils, log_utils

tf.enable_eager_execution(**utils.tf_session_config())

"""
Trajectory plot of interesting episodes (unnecessary annotation removed)
"""

linewidth = 3.5

area = 1

if area == 6:
    end_to_end_dir = './area6/end_to_end/session_2019-01-30_14-04-20/rgb_resnet50_nn_control_simulator/trajectories'
    waypoint_dir = './area6/waypoint/session_2019-01-30_13-57-31/rgb_resnet50_nn_waypoint_simulator/trajectories'

    interesting_goals = [40, 53, 56, 82]
    plotting_steps = {40: 6, 53: 6.5, 56: 5, 82: 6}
    quiver_freq = {40: 30, 53: 30, 56: 30, 82: 30}
    goal_locations = {40: [13.4, 49.6], 53: [16.5, 34.85], 56: [11.5, 7.00], 82: [7.35, 12.65]}
    # interesting_goals = [40]
    # plotting_steps = {40: 6}
    # goal_locations = {40: [13.4, 49.6]}
    # planner_data_idx = {40: 11}
    # quiver_freq = {40: 10}
elif area == 1:
    margin0 = 0.3
    margin1 = 0.5

    # end_to_end_dir = './area1/end_to_end/session_2019-01-30_13-55-54/rgb_resnet50_nn_control_simulator/trajectories'
    # waypoint_dir = './area1/waypoint/session_2019-01-30_13-47-21/rgb_resnet50_nn_waypoint_simulator/trajectories'
    # interesting_goals = [31, 38, 73]
    # plotting_steps = {31: 5.5, 38: 6.0, 73:5}
    # quiver_freq = {31: 25, 38: 25, 73:25}
    # goal_locations = {31: [12.5, 24.7], 38: [13.15, 22.95], 73: [12.70, 25.15]}

    # Baseline
    # end_to_end_dir = '/media/anjianl/My Book/WayPtNav/paper_results/single_shot_0.25/session_2019-12-05_19-33-29_end2end_pretrain_0.25/rgb_resnet50_nn_control_simulator/trajectories'
    # end_to_end_dir = '/media/anjianl/My Book/WayPtNav/paper_results/single_shot_0.25/session_2019-09-19_10-08-01_pretrained_ctrlhorizon_0.25/rgb_resnet50_nn_waypoint_simulator/trajectories'
    # end_to_end_dir = '/media/anjianl/My Book/WayPtNav/paper_results/single_shot_0.25/session_2019-12-05_18-11-08_reachability_no_disturbance_0.25/rgb_resnet50_nn_waypoint_simulator/trajectories'

    # oldcost expert
    # end_to_end_dir = '/media/anjianl/My Book/WayPtNav/paper_results/expert/session_2019-12-12_17-43-53_oldcost_expert_area5a_20maps/expert_simulator/trajectories'

    # new expert without disturbances
    end_to_end_dir = '/media/anjianl/My Book/WayPtNav/paper_results/expert/session_2019-12-12_18-07-16_reachability_no_dist_expert_area5a_6maps/expert_simulator/trajectories'

    # Testing
    # waypoint_dir = '/media/anjianl/My Book/WayPtNav/paper_results/single_shot_0.25/session_2019-12-04_18-15-33_end-to-end_reachability_ctrlhorizon_0.25/rgb_resnet50_nn_control_simulator/trajectories'
    # waypoint_dir = '/media/anjianl/My Book/WayPtNav/paper_results/single_shot_0.25/session_2019-12-03_16-10-17_lr-4_reg-6_ckpt19_v2_ctrlhorizon_0.25/rgb_resnet50_nn_waypoint_simulator/trajectories'
    waypoint_dir = '/media/anjianl/My Book/WayPtNav/paper_results/expert/session_2019-12-12_17-30-03_reachability_expert_area5a_20maps/expert_simulator/trajectories'

    # no disturbances map 1
    # interesting_goals = [1]
    # plotting_steps = {1: 5}
    # quiver_freq = {1: 25}
    # goal_locations = {1: [7.90, 18.95]}

    # # map 6
    # interesting_goals = [6]
    # plotting_steps = {6: 5}
    # quiver_freq = {6: 25}
    # goal_locations = {6: [8.20, 26.75]}

    # # map 10
    # interesting_goals = [10]
    # plotting_steps = {10: 5}
    # quiver_freq = {10: 25}
    # goal_locations = {10: [8.70, 44.40]}

    # # WayPtNav
    # interesting_goals = [57]
    # plotting_steps = {57: 5}
    # quiver_freq = {57: 50}
    # goal_locations = {57: [5.45, 10.80]}

    # expert old vs our
    # interesting_goals = [0]
    # plotting_steps = {0: 5}
    # quiver_freq = {0: 50}
    # goal_locations = {0: [31.55, 30.55]}

    # expert disturbances vs no dist
    # interesting_goals = [1]
    # plotting_steps = {1: 5}
    # quiver_freq = {1: 50}
    # goal_locations = {1: [38.00, 10.75]}

    # very big plot expert
    interesting_goals = [2]
    plotting_steps = {2: 5}
    quiver_freq = {2: 50}
    goal_locations = {2: [8.80, 33.10]}

    # e2e learning
    # interesting_goals = [10]
    # plotting_steps = {10: 5}
    # quiver_freq = {10: 50}
    # goal_locations = {10: [8.70, 44.40]}


else:
    assert False


def plot_top_view(filename, mode, file_number, simulator):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    # Plot Map
    ax.imshow(data['occupancy_grid'], cmap='gray_r', extent=data['map_bounds_extent'],
              vmax=1.5, vmin=-.5, origin='lower')

    # Plot hard margin
    # render_margin(ax, margin=margin0, alpha=.5, simulator=simulator)
    # render_margin(ax, margin=margin1, alpha=.35, simulator=simulator)


    pos_k2 = data['vehicle_trajectory']['position_nk2'][0]

    # Plot Trajectory
    ax.plot(pos_k2[:, 0], pos_k2[:, 1], 'r-', linewidth=linewidth, zorder=1)

    # Plot Waypoints Or Quiver
    if 'waypoint_config' in data['vehicle_data'].keys():
        system_pos_n2 = data['vehicle_data']['system_config']['position_nk2'][:, 0]
        system_heading_n1 = data['vehicle_data']['system_config']['heading_nk1'][:, 0]
        ax.quiver(system_pos_n2[:, 0][::2], system_pos_n2[:, 1][::2], np.cos(system_heading_n1[::2]),
                  np.sin(system_heading_n1[::2]), zorder=5)
    else:
        heading_k1 = data['vehicle_trajectory']['heading_nk1'][0]
        ax.quiver(pos_k2[:, 0][::quiver_freq[file_number]], pos_k2[:, 1][::quiver_freq[file_number]],
                  np.cos(heading_k1[::quiver_freq[file_number]]), np.sin(heading_k1[::quiver_freq[file_number]]),
                  zorder=5)

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
    ax.set_xlim(min_x - 0.5 * delta_x, max_x + 0.5 * delta_x)
    ax.set_ylim(min_y - 0.5 * delta_y, max_y + 0.5 * delta_y)
    # ax.set_xlim(start_2[0]-delta, start_2[0]+delta)
    # ax.set_ylim(start_2[1]-delta, start_2[1]+delta)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the plot
    # fig.savefig('./plots/area{:d}_traj_{:d}_{:s}.pdf'.format(area, file_number, mode), bbox_inches='tight', dpi=300)
    # fig.savefig('~/Desktop/area{:d}_traj_{:d}_{:s}.png'.format(area, file_number, mode), bbox_inches='tight')


def plot_interesting_episodes(mode, simulator):
    if mode == 'end_to_end':
        directory_name = end_to_end_dir
    elif mode == 'waypoint':
        directory_name = waypoint_dir
    else:
        assert False

    assert (os.path.exists(directory_name))
    data_files = os.listdir(directory_name)
    data_files = list(filter(lambda x: 'metadata' not in x, data_files))

    for data_file in data_files:
        file_number = int(data_file.split('.pkl')[0][5:])
        if file_number in interesting_goals:
            delta = plotting_steps[file_number]
            plot_top_view(os.path.join(directory_name, data_file), mode, file_number, simulator)

def get_simulator():

        parser = argparse.ArgumentParser(description='Process the command line inputs')
        parser.add_argument("-p", "--params", required=True, help='the path to the parameter file')
        args = parser.parse_args()

        p = create_params(args.params)

        p.simulator_params = p.data_creation.simulator_params
        p.simulator_params.simulator.parse_params(p.simulator_params)

        simulator = p.simulator_params.simulator(p.simulator_params)

        return simulator

def create_params(param_file):
        """
        Create the parameters given the path of the parameter file.
        """
        # Execute this if python > 3.4
        try:
            spec = importlib.util.spec_from_file_location('parameter_loader', param_file)
            foo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(foo)
        except AttributeError:
            # Execute this if python = 2.7 (i.e. when running on real robot with ROS)
            module_name = param_file.replace('/', '.').replace('.py', '')
            foo = importlib.import_module(module_name)
        return foo.create_params()

def render_margin(ax, margin, alpha, simulator, color=None):
    """
    Render a margin around the occupied space indicating the intensity
    of the obstacle avoidance cost function.
    """
    y_dim, x_dim =simulator.obstacle_map.occupancy_grid_map.shape
    xs = np.linspace(simulator.obstacle_map.map_bounds[0][0], simulator.obstacle_map.map_bounds[1][0], x_dim)
    ys = np.linspace(simulator.obstacle_map.map_bounds[0][1], simulator.obstacle_map.map_bounds[1][1], y_dim)
    xs, ys = np.meshgrid(xs, ys)
    xs = xs.ravel()
    ys = ys.ravel()
    pos_n12 = np.stack([xs, ys], axis=1)[:, None]
    dists_nk = simulator.obstacle_map.dist_to_nearest_obs(pos_n12).numpy()

    # margin_mask_n = np.logical_and((dists_nk < margin)[:, 0], (dists_nk > 0.1)[:, 0])
    margin_mask_n = (dists_nk < margin)[:, 0]
    margin_mask_mn = margin_mask_n.reshape(simulator.obstacle_map.occupancy_grid_map.shape)
    mask = np.logical_and(simulator.obstacle_map.occupancy_grid_map, margin_mask_mn == 0)

    margin_img = np.ma.masked_where(mask, margin_mask_mn)

    if color == None:
        ax.imshow(margin_img, cmap='gray_r',
                  extent=np.array(simulator.obstacle_map.map_bounds).flatten(order='F'),
                  origin='lower', alpha=alpha, vmax=2.0)
    else:
        ax.imshow(margin_img, cmap=color,
                  extent=np.array(simulator.obstacle_map.map_bounds).flatten(order='F'),
                  origin='lower', alpha=alpha, vmax=2.0)


if __name__ == '__main__':
    simulator = get_simulator()
    matplotlib.style.use('ggplot')
    plot_interesting_episodes(mode='waypoint', simulator=simulator)
    plot_interesting_episodes(mode='end_to_end', simulator=simulator)
    plt.show()
