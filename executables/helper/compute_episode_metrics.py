import numpy as np
import os
import pickle
from colorama import Fore, Style

def get_session_dirs(exp_name):
    """
    For a given experiment returns the succesfull
    sessions for each method.
    """
    if exp_name == 'Dynamic Obstacle Experiment':
        session_dirs = {'E2E': [],
                        'Freespace': [],
                        'Map': [],
                        'WayPtNav': ["/home/ext_drive/somilb/data/experiments/CORL_2019/final/Dynamic Obstacle Experiment/WayPtNav/session_2019-07-02_20-44-13/rgb_resnet50_nn_waypoint_simulator/trajectories"]}
    elif exp_name == 'Leave Vicon Room':
        session_dirs = {'E2E' : [],
                        'Freespace': [],
                        'Map': ['/home/ext_drive/somilb/data/experiments/CORL_2019/final/Leave Vicon Room/Map Based Planning/session_2019-07-02_20-19-48/classical_simulator/trajectories',
                                '/home/ext_drive/somilb/data/experiments/CORL_2019/final/Leave Vicon Room/Map Based Planning/other_sessions/session_2019-07-03_14-13-38/classical_simulator/trajectories',
                                '/home/ext_drive/somilb/data/experiments/CORL_2019/final/Leave Vicon Room/Map Based Planning/other_sessions/session_2019-07-03_14-15-05/classical_simulator/trajectories',
                                '/home/ext_drive/somilb/data/experiments/CORL_2019/final/Leave Vicon Room/Map Based Planning/other_sessions/session_2019-07-03_14-16-51/classical_simulator/trajectories',
                                '/home/ext_drive/somilb/data/experiments/CORL_2019/final/Leave Vicon Room/Map Based Planning/other_sessions/session_2019-07-03_14-18-55/classical_simulator/trajectories',
                                ],
                        'WayPtNav': ['/home/ext_drive/somilb/data/experiments/CORL_2019/final/Leave Vicon Room/WayptNav/session_2019-07-02_20-22-16/rgb_resnet50_nn_waypoint_simulator/trajectories',
                                     '/home/ext_drive/somilb/data/experiments/CORL_2019/final/Leave Vicon Room/WayptNav/session_2019-07-02_20-24-32/rgb_resnet50_nn_waypoint_simulator/trajectories',
                                     '/home/ext_drive/somilb/data/experiments/CORL_2019/final/Leave Vicon Room/WayptNav/session_2019-07-05_12-59-27/rgb_resnet50_nn_waypoint_simulator/trajectories',
                                     '/home/ext_drive/somilb/data/experiments/CORL_2019/final/Leave Vicon Room/WayptNav/session_2019-07-05_15-09-48/rgb_resnet50_nn_waypoint_simulator/trajectories',
                                     '/home/ext_drive/somilb/data/experiments/CORL_2019/final/Leave Vicon Room/WayptNav/session_2019-07-05_15-11-06/rgb_resnet50_nn_waypoint_simulator/trajectories'
                                    ]}
    elif exp_name == 'SDH7 Bikes':
        session_dirs = {'E2E': ['/home/ext_drive/somilb/data/experiments/CORL_2019/final/SDH7 Bikes/E2E/session_2019-07-03_21-58-09/rgb_resnet50_nn_control_simulator/clipped_trajectories',
                               '/home/ext_drive/somilb/data/experiments/CORL_2019/final/SDH7 Bikes/E2E/session_2019-07-03_21-59-18/rgb_resnet50_nn_control_simulator/clipped_trajectories'],
                        'Freespace': [],
                        'Map': [],
                        'WayPtNav': ['/home/ext_drive/somilb/data/experiments/CORL_2019/final/SDH7 Bikes/WayPtNav/session_2019-07-03_21-42-52/rgb_resnet50_nn_waypoint_simulator/trajectories',
                                     '/home/ext_drive/somilb/data/experiments/CORL_2019/final/SDH7 Bikes/WayPtNav/session_2019-07-03_21-44-20/rgb_resnet50_nn_waypoint_simulator/trajectories',
                                     '/home/ext_drive/somilb/data/experiments/CORL_2019/final/SDH7 Bikes/WayPtNav/session_2019-07-03_21-51-48/rgb_resnet50_nn_waypoint_simulator/trajectories']}
    elif exp_name == 'SDH7 Original Experiment':
        session_dirs = {'E2E': ['/home/ext_drive/somilb/data/experiments/CORL_2019/final/SDH7 Original Experiment/E2E/session_2019-07-03_20-54-24/rgb_resnet50_nn_control_simulator/clipped_trajectories',
                                '/home/ext_drive/somilb/data/experiments/CORL_2019/final/SDH7 Original Experiment/E2E/session_2019-07-03_20-57-04/rgb_resnet50_nn_control_simulator/clipped_trajectories'],
                        'Freespace': [],
                        'Map': ['/home/ext_drive/somilb/data/experiments/CORL_2019/final/SDH7 Original Experiment/Map Based Planning/success/session_2019-07-03_21-01-42/classical_simulator/trajectories',
                                '/home/ext_drive/somilb/data/experiments/CORL_2019/final/SDH7 Original Experiment/Map Based Planning/success/session_2019-07-03_21-10-03/classical_simulator/trajectories',
                                '/home/ext_drive/somilb/data/experiments/CORL_2019/final/SDH7 Original Experiment/Map Based Planning/success/session_2019-07-03_21-11-48/classical_simulator/trajectories'],
                        'WayPtNav': ['/home/ext_drive/somilb/data/experiments/CORL_2019/final/SDH7 Original Experiment/WayPtNav/session_2019-07-03_20-45-45/rgb_resnet50_nn_waypoint_simulator/trajectories',
                                     '/home/ext_drive/somilb/data/experiments/CORL_2019/final/SDH7 Original Experiment/WayPtNav/session_2019-07-03_20-47-25/rgb_resnet50_nn_waypoint_simulator/trajectories',
                                     '/home/ext_drive/somilb/data/experiments/CORL_2019/final/SDH7 Original Experiment/WayPtNav/session_2019-07-03_20-50-01/rgb_resnet50_nn_waypoint_simulator/trajectories']
                       }
    elif exp_name == 'Sunny SDH7 Experiment':
        session_dirs = {'E2E': [],
                        'Freespace': [],
                        'Map': [],
                        'WayPtNav': ['/home/ext_drive/somilb/data/experiments/CORL_2019/final/Sunny SDH7 Experiment/WayPtNav/session_2019-07-03_18-02-29/rgb_resnet50_nn_waypoint_simulator/trajectories',
                                     '/home/ext_drive/somilb/data/experiments/CORL_2019/final/Sunny SDH7 Experiment/WayPtNav/session_2019-07-03_18-07-57/rgb_resnet50_nn_waypoint_simulator/trajectories']
                       }
    elif exp_name == 'Debug Large Jerk':
        session_dirs = {'Waypt_7k_Grid': ['/home/ext_drive/somilb/data/experiments/CORL_2019/final/Leave_Vicon_debug_jerk/waypt_grid_7k_pts/session_2019-07-05_12-59-27/rgb_resnet50_nn_waypoint_simulator/trajectories'],
                        'Waypt_20k_Grid': ['/home/ext_drive/somilb/data/experiments/CORL_2019/final/Leave_Vicon_debug_jerk/waypt_grid_20k_pts/session_2019-07-05_13-01-50/rgb_resnet50_nn_waypoint_simulator/trajectories'],
                        'Waypt_20k_Grid_commanded': ['/home/ext_drive/somilb/data/experiments/CORL_2019/final/Leave_Vicon_debug_jerk/waypt_grid_20k_pts/use_commanded_velocity/session_2019-07-05_13-42-36/rgb_resnet50_nn_waypoint_simulator/trajectories']}
    else:
        raise NotImplementedError
    return session_dirs

def compute_metric_for_goal_numbers(directory_name, goal_numbers, metric_name, remove_outliers):
    assert(os.path.exists(directory_name))
    data_files = os.listdir(directory_name)
    data_files = list(filter(lambda x: 'metadata' not in x, data_files))
    metric_vals = []
    for data_file in data_files:
        if '.pkl' in data_file:
            file_number = int(data_file.split('.pkl')[0][5:])
            if file_number in goal_numbers:
                if metric_name == 'episode_length':
                    metric_vals.append(get_episode_length(os.path.join(directory_name, data_file)))
                elif metric_name == 'avg_acceleration':
                    metric_vals.append(get_avg_episode_acceleration(os.path.join(directory_name, data_file), remove_outliers))
                elif metric_name == 'avg_jerk':
                    metric_vals.append(get_avg_episode_jerk(os.path.join(directory_name, data_file), remove_outliers))
                else:
                    assert False
    return metric_vals

def get_episode_length(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['vehicle_trajectory']['k']*.05


def reject_outliers(data, m=2):
    """
    From Here
    https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
    """
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/(mdev if mdev else 1.)
    return data[s<m]


def get_avg_episode_acceleration(filename, remove_outliers):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    episode_accelerations = np.diff(data['vehicle_trajectory']['speed_nk1'][0, :, 0])/.05
    if remove_outliers:
        episode_accelerations = reject_outliers(episode_accelerations)
    avg_acceleration_magnitude = np.mean(np.abs(episode_accelerations))
    return avg_acceleration_magnitude

def get_avg_episode_jerk(filename, remove_outliers):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    episode_accelerations = np.diff(data['vehicle_trajectory']['speed_nk1'][0, :, 0])/.05
    episode_jerks = np.diff(episode_accelerations)/.05
    if remove_outliers:
        episode_jerks = reject_outliers(episode_jerks)
    avg_jerk_magnitude = np.mean(np.abs(episode_jerks))

    #import matplotlib
    #matplotlib.use('Agg')
    #import matplotlib.pyplot as plt
    #plt.style.use('ggplot')
    #fig = plt.figure(figsize=(20,20))
    #ax = fig.add_subplot(221)
    #ax.plot(data['vehicle_trajectory']['speed_nk1'][0, :, 0])
    #ax.set_title('Velocity (m/s)')

    #ax = fig.add_subplot(222)
    #ax.plot(episode_accelerations)
    #ax.set_title('Acceleration (m/s^2)')

    #ax = fig.add_subplot(223)
    #ax.plot(episode_jerks)
    #ax.set_title('Jerk (m/s^3)')

    #ax = fig.add_subplot(224)
    #ax.plot(episode_jerks_no_outliers)
    #ax.set_title('Jerk No Outliers (m/s^3)')
    #import pdb; pdb.set_trace()

    #fig.savefig('./debug_large_jerk.png', bbox_inches='tight')
    return avg_jerk_magnitude

def compute_experiment_metrics():
    exp_names = ['Dynamic Obstacle Experiment', 'Leave Vicon Room', 'SDH7 Bikes', 'SDH7 Original Experiment', 'Sunny SDH7 Experiment']
    #exp_names = ['Debug Large Jerk']
    for exp_name in exp_names:
        session_dirs = get_session_dirs(exp_name)
        #methods = ['Waypt_7k_Grid', 'Waypt_20k_Grid', 'Waypt_20k_Grid_commanded']
        methods = ['E2E', 'Freespace', 'Map', 'WayPtNav']
        print(Fore.GREEN + '{:s}\n'.format(exp_name) + Style.RESET_ALL)
        for method in methods:
            jerks = []
            accs = []
            lengths = []
            
            # Dont remove outliers for the E2E method as E2E is
            # expected to have a lot of big outliers 
            remove_outliers = (method != 'E2E')
            for trajectory_dir in session_dirs[method]:
                jerks.append(compute_metric_for_goal_numbers(trajectory_dir, [0], 'avg_jerk',
                                                             remove_outliers=remove_outliers)[0])
                accs.append(compute_metric_for_goal_numbers(trajectory_dir, [0],
                                                            'avg_acceleration',
                                                            remove_outliers=remove_outliers)[0])
                lengths.append(compute_metric_for_goal_numbers(trajectory_dir, [0],
                                                               'episode_length',
                                                               remove_outliers=remove_outliers)[0])
            if len(session_dirs[method]) > 0:
                to_print = Fore.RED + '{:s}\n'.format(method) + Style.RESET_ALL
                to_print += 'Episode Length: Mean- {:.3f} Std Dev- {:.3f}\n'.format(np.mean(lengths),
                                                                                  np.std(lengths))
                to_print += 'Acceleration: Mean- {:.3f} Std Dev- {:.3f}\n'.format(np.mean(accs),
                                                                                np.std(accs))
                to_print += 'Jerk: Mean- {:.3f} Std Dev- {:.3f}\n'.format(np.mean(jerks),
                                                                        np.std(jerks))
                to_print += '\n'
                print(to_print)

if __name__ == '__main__':
    compute_experiment_metrics()