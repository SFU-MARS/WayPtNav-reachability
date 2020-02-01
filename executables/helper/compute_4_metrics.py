import os
import matplotlib

matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pickle

"""
Compute success rate, time to reach, acceleration and jerk from trajectory files.
"""

# reachability end-to-end
end_to_end_dir = '/media/anjianl/My Book/WayPtNav/paper_results/single_shot_0.25/session_2019-12-04_18-15-33_end-to-end_reachability_ctrlhorizon_0.25/rgb_resnet50_nn_control_simulator/trajectories'

# reachability waypoint
waypoint_dir = '/media/anjianl/My Book/WayPtNav/paper_results/single_shot_0.25/session_2019-12-03_16-10-17_lr-4_reg-6_ckpt19_v2_ctrlhorizon_0.25/rgb_resnet50_nn_waypoint_simulator/trajectories'

# expert_dir = '/media/anjianl/My Book/WayPtNav/paper_results/single_shot_0.25/session_2019-12-03_16-10-17_lr-4_reg-6_ckpt19_v2_ctrlhorizon_0.25/rgb_resnet50_nn_waypoint_simulator/trajectories'

expert_dir = '/media/anjianl/My Book/WayPtNav/paper_results/single_shot_0.25/session_2019-12-05_19-33-29_end2end_pretrain_0.25/rgb_resnet50_nn_control_simulator/trajectories'

# oldcost waypt
mapper_dir = '/media/anjianl/My Book/WayPtNav/paper_results/single_shot_0.25/session_2019-09-19_10-08-01_pretrained_ctrlhorizon_0.25/rgb_resnet50_nn_waypoint_simulator/trajectories'

# reachability no disturbances
reactive_dir = '/home/anjianl/Desktop/project/WayPtNav/reproduce_WayptNavResults_v3/session_2019-12-05_18-11-08_reachability_no_disturbance_0.25/rgb_resnet50_nn_waypoint_simulator/trajectories'

bar_fontsize = 10
xticks_fontsize = 14

waypoint_color = 'r'
end_to_end_color = 'gray'
expert_color = '#4885ea'
waypoint_alpha = .8
end_to_end_alpha = .8
expert_alpha = .8

waypoint_method_name = 'WayPtNav (our)'


def get_episodes_percentages(directory_name):
    assert (os.path.exists(directory_name))
    metadata_file = os.path.join(directory_name, 'metadata.pkl')
    assert (os.path.exists(metadata_file))
    with open(metadata_file, 'rb') as f:
        data = pickle.load(f)
    episode_types = np.array(data['episode_type_string'])
    n = len(episode_types)
    percents = {}
    reasons = ['Success', 'Collision', 'Timeout']
    for reason in reasons:
        percents[reason] = len(np.where(episode_types == reason)[0]) / n
    return percents


def get_goal_numbers_for_episode_type(directory_name, reason='Success'):
    assert (os.path.exists(directory_name))
    metadata_file = os.path.join(directory_name, 'metadata.pkl')
    assert (os.path.exists(metadata_file))
    with open(metadata_file, 'rb') as f:
        data = pickle.load(f)

    episode_types = np.array(data['episode_type_string'])
    idxs = np.where(episode_types == reason)[0]
    return np.array(data['episode_number'])[idxs]


def compute_metric_for_goal_numbers(directory_name, goal_numbers, metric_name):
    assert (os.path.exists(directory_name))
    data_files = os.listdir(directory_name)
    data_files = list(filter(lambda x: 'metadata' not in x, data_files))
    metric_vals = []
    for data_file in data_files:
        file_number = int(data_file.split('.pkl')[0][5:])
        if file_number in goal_numbers:
            if metric_name == 'episode_length':
                metric_vals.append(get_episode_length(os.path.join(directory_name, data_file)))
            elif metric_name == 'avg_acceleration':
                metric_vals.append(get_avg_episode_acceleration(os.path.join(directory_name, data_file)))
            elif metric_name == 'avg_jerk':
                metric_vals.append(get_avg_episode_jerk(os.path.join(directory_name, data_file)))
            else:
                assert False
    return metric_vals


def get_episode_length(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['vehicle_trajectory']['k'] * .05


def get_avg_episode_acceleration(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    episode_accelerations = np.diff(data['vehicle_trajectory']['speed_nk1'][0, :, 0]) / .05
    avg_acceleration_magnitude = np.mean(np.abs(episode_accelerations))
    return avg_acceleration_magnitude


def get_avg_episode_jerk(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    episode_accelerations = np.diff(data['vehicle_trajectory']['speed_nk1'][0, :, 0]) / .05
    episode_jerks = np.diff(episode_accelerations) / .05
    avg_jerk_magnitude = np.mean(np.abs(episode_jerks))
    return avg_jerk_magnitude


def compute_set_intersection(list1, list2):
    return list(set(list1) & set(list2))


def get_bar_chart_data(return_expert_data=False):
    # Get Overall Success, Collision, Timeout Rate
    end_to_end_percentages = get_episodes_percentages(end_to_end_dir)
    waypoint_percentages = get_episodes_percentages(waypoint_dir)

    # Compute the goals over which waypoint and end to end were both successful
    end_to_end_success_goals = get_goal_numbers_for_episode_type(end_to_end_dir)
    waypoint_success_goals = get_goal_numbers_for_episode_type(waypoint_dir)
    common_success_goals = compute_set_intersection(end_to_end_success_goals, waypoint_success_goals)

    # Get the episode lengths for this commmon set of goals
    end_to_end_common_episode_lengths = compute_metric_for_goal_numbers(end_to_end_dir, common_success_goals,
                                                                        'episode_length')
    waypoint_common_episode_lengths = compute_metric_for_goal_numbers(waypoint_dir, common_success_goals,
                                                                      'episode_length')

    # Compute the Average Acceleration Magnitude for this common set of goals
    end_to_end_common_avg_accelerations = compute_metric_for_goal_numbers(end_to_end_dir, common_success_goals,
                                                                          'avg_acceleration')
    waypoint_common_avg_accelerations = compute_metric_for_goal_numbers(waypoint_dir, common_success_goals,
                                                                        'avg_acceleration')

    # Compute the Average Jerk Magnitude for this common set of goals
    end_to_end_common_avg_jerks = compute_metric_for_goal_numbers(end_to_end_dir, common_success_goals, 'avg_jerk')
    waypoint_common_avg_jerks = compute_metric_for_goal_numbers(waypoint_dir, common_success_goals, 'avg_jerk')

    e2e_data = [end_to_end_percentages, end_to_end_common_episode_lengths, end_to_end_common_avg_accelerations,
                end_to_end_common_avg_jerks]
    waypt_data = [waypoint_percentages, waypoint_success_goals, waypoint_common_episode_lengths,
                  waypoint_common_avg_accelerations, waypoint_common_avg_jerks]

    if return_expert_data:
        # expert_percentages = get_episodes_percentages(expert_dir)
        expert_percentages = {'Success': 1.0,
                              'Collision': 0.0,
                              'Timeout': 0.0}
        expert_common_episode_lengths = compute_metric_for_goal_numbers(expert_dir, common_success_goals,
                                                                        'episode_length')
        expert_common_avg_accelerations = compute_metric_for_goal_numbers(expert_dir, common_success_goals,
                                                                          'avg_acceleration')
        expert_common_avg_jerks = compute_metric_for_goal_numbers(expert_dir, common_success_goals, 'avg_jerk')
        expert_data = [expert_percentages, expert_common_episode_lengths, expert_common_avg_accelerations,
                       expert_common_avg_jerks]
        return e2e_data, waypt_data, expert_data
    else:
        return e2e_data, waypt_data


def get_bar_chart_data_v2(return_expert_data=False):
    # Get Overall Success, Collision, Timeout Rate
    end_to_end_percentages = get_episodes_percentages(end_to_end_dir)
    waypoint_percentages = get_episodes_percentages(waypoint_dir)
    mapper_percentages = get_episodes_percentages(mapper_dir)
    reactive_percentages = get_episodes_percentages(reactive_dir)

    # Compute the goals over which all methods were successful
    end_to_end_success_goals = get_goal_numbers_for_episode_type(end_to_end_dir)
    waypoint_success_goals = get_goal_numbers_for_episode_type(waypoint_dir)
    mapper_success_goals = get_goal_numbers_for_episode_type(mapper_dir)
    reactive_success_goals = get_goal_numbers_for_episode_type(reactive_dir)

    common_success_goals = compute_set_intersection(end_to_end_success_goals, waypoint_success_goals)
    common_success_goals = compute_set_intersection(common_success_goals, mapper_success_goals)
    common_success_goals = compute_set_intersection(common_success_goals, reactive_success_goals)

    # Get the episode lengths for this commmon set of goals
    end_to_end_common_episode_lengths = compute_metric_for_goal_numbers(end_to_end_dir, common_success_goals,
                                                                        'episode_length')
    waypoint_common_episode_lengths = compute_metric_for_goal_numbers(waypoint_dir, common_success_goals,
                                                                      'episode_length')
    mapper_common_episode_lengths = compute_metric_for_goal_numbers(mapper_dir, common_success_goals,
                                                                    'episode_length')
    reactive_common_episode_lengths = compute_metric_for_goal_numbers(reactive_dir, common_success_goals,
                                                                      'episode_length')

    # Compute the Average Acceleration Magnitude for this common set of goals
    end_to_end_common_avg_accelerations = compute_metric_for_goal_numbers(end_to_end_dir, common_success_goals,
                                                                          'avg_acceleration')
    waypoint_common_avg_accelerations = compute_metric_for_goal_numbers(waypoint_dir, common_success_goals,
                                                                        'avg_acceleration')
    mapper_common_avg_accelerations = compute_metric_for_goal_numbers(mapper_dir, common_success_goals,
                                                                      'avg_acceleration')
    reactive_common_avg_accelerations = compute_metric_for_goal_numbers(reactive_dir, common_success_goals,
                                                                        'avg_acceleration')

    # Compute the Average Jerk Magnitude for this common set of goals
    end_to_end_common_avg_jerks = compute_metric_for_goal_numbers(end_to_end_dir, common_success_goals, 'avg_jerk')
    waypoint_common_avg_jerks = compute_metric_for_goal_numbers(waypoint_dir, common_success_goals, 'avg_jerk')
    mapper_common_avg_jerks = compute_metric_for_goal_numbers(mapper_dir, common_success_goals, 'avg_jerk')
    reactive_common_avg_jerks = compute_metric_for_goal_numbers(reactive_dir, common_success_goals, 'avg_jerk')

    e2e_data = [end_to_end_percentages, end_to_end_common_episode_lengths, end_to_end_common_avg_accelerations,
                end_to_end_common_avg_jerks]
    waypt_data = [waypoint_percentages, waypoint_success_goals, waypoint_common_episode_lengths,
                  waypoint_common_avg_accelerations, waypoint_common_avg_jerks]
    mapper_data = [mapper_percentages, mapper_success_goals, mapper_common_episode_lengths,
                   mapper_common_avg_accelerations, mapper_common_avg_jerks]
    reactive_data = [reactive_percentages, reactive_success_goals, reactive_common_episode_lengths,
                     reactive_common_avg_accelerations, reactive_common_avg_jerks]

    if return_expert_data:
        # expert_percentages = get_episodes_percentages(expert_dir)
        expert_percentages = {'Success': 1.0,
                              'Collision': 0.0,
                              'Timeout': 0.0}
        expert_common_episode_lengths = compute_metric_for_goal_numbers(expert_dir, common_success_goals,
                                                                        'episode_length')
        expert_common_avg_accelerations = compute_metric_for_goal_numbers(expert_dir, common_success_goals,
                                                                          'avg_acceleration')
        expert_common_avg_jerks = compute_metric_for_goal_numbers(expert_dir, common_success_goals, 'avg_jerk')
        expert_data = [expert_percentages, expert_common_episode_lengths, expert_common_avg_accelerations,
                       expert_common_avg_jerks]
        return e2e_data, waypt_data, mapper_data, reactive_data, expert_data
    else:
        return e2e_data, waypt_data, mapper_data, reactive_data


def make_bar_chart_v0():
    end_to_end_data, waypoint_data = get_bar_chart_data()

    end_to_end_percentages, end_to_end_common_episode_lengths, end_to_end_common_avg_accelerations, end_to_end_common_avg_jerks = end_to_end_data
    waypoint_percentages, waypoint_success_goals, waypoint_common_episode_lengths, waypoint_common_avg_accelerations, waypoint_common_avg_jerks = waypoint_data

    # Plot Stuff
    matplotlib.style.use('ggplot')
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)

    bar_width = .3

    end_to_end_data = [end_to_end_percentages['Success'] * 100.0, np.mean(end_to_end_common_episode_lengths),
                       np.mean(end_to_end_common_avg_accelerations), np.mean(end_to_end_common_avg_jerks)]

    waypoint_data = [waypoint_percentages['Success'] * 100.0, np.mean(waypoint_common_episode_lengths),
                     np.mean(waypoint_common_avg_accelerations), np.mean(waypoint_common_avg_jerks)]

    end_to_end_data = np.array(end_to_end_data)
    waypoint_data = np.array(waypoint_data)

    scale_factor = np.maximum(end_to_end_data, waypoint_data)

    idx = np.r_[:4]

    rects_end_to_end = ax.bar(idx, end_to_end_data / scale_factor, bar_width, color='r', alpha=.8, label='End To End')
    rects_waypoints = ax.bar(idx + bar_width, waypoint_data / scale_factor, bar_width, color='b', alpha=.8,
                             label='Waypoint')

    plt.xticks(idx + bar_width / 2.,
               ['% Success', 'Average Time to Reach Goal', 'Average Acceleration', 'Average Jerk'],
               fontsize=xticks_fontsize)

    units = ['%', 's', r'm/s$^2$', r'm/s$^3$']

    for rect, datum, unit in zip(rects_end_to_end, end_to_end_data, units):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.01 * height, '{:.2f} {:s}'.format(datum, unit),
                ha='center', va='bottom', fontsize=bar_fontsize)

    for rect, datum, unit in zip(rects_waypoints, waypoint_data, units):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.01 * height, '{:.2f} {:s}'.format(datum, unit),
                ha='center', va='bottom', fontsize=bar_fontsize)

    ax.set_yticklabels([])
    ax.legend(loc='upper right', prop={'size': 14}, ncol=2)
    plt.savefig('./plots/bar_chart.pdf', bbox_inches='tight')


def make_bar_chart_v1(x_label_fontsize=12, bar_fontsize=12, legend_size=14, error_bars_mode='std'):
    end_to_end_data, waypoint_data = get_bar_chart_data()

    end_to_end_percentages, end_to_end_common_episode_lengths, end_to_end_common_avg_accelerations, end_to_end_common_avg_jerks = end_to_end_data
    waypoint_percentages, waypoint_success_goals, waypoint_common_episode_lengths, waypoint_common_avg_accelerations, waypoint_common_avg_jerks = waypoint_data

    # Plot Stuff
    # matplotlib.style.use('ggplot')
    fig = plt.figure(figsize=(12, 5))
    # ax = fig.add_subplot(141)

    bar_width = .3

    end_to_end_data = [end_to_end_percentages['Success'] * 100.0, np.mean(end_to_end_common_episode_lengths),
                       np.mean(end_to_end_common_avg_accelerations), np.mean(end_to_end_common_avg_jerks)]

    waypoint_data = [waypoint_percentages['Success'] * 100.0, np.mean(waypoint_common_episode_lengths),
                     np.mean(waypoint_common_avg_accelerations), np.mean(waypoint_common_avg_jerks)]

    if error_bars_mode == 'std':
        end_to_end_error_bars = [None, np.std(end_to_end_common_episode_lengths),
                                 np.std(end_to_end_common_avg_accelerations), np.std(end_to_end_common_avg_jerks)]
        waypoint_error_bars = [None, np.std(waypoint_common_episode_lengths),
                               np.std(waypoint_common_avg_accelerations), np.std(waypoint_common_avg_jerks)]
    elif error_bars_mode == 'iqr':
        end_to_end_error_bars = [None, np.std(end_to_end_common_episode_lengths),
                                 np.std(end_to_end_common_avg_accelerations), np.std(end_to_end_common_avg_jerks)]
        waypoint_error_bars = [None, np.std(waypoint_common_episode_lengths),
                               np.std(waypoint_common_avg_accelerations), np.std(waypoint_common_avg_jerks)]
    else:
        assert False

    end_to_end_data = np.array(end_to_end_data)
    waypoint_data = np.array(waypoint_data)

    x_labels = ['Success (%)\n(A)', 'Average Time to Reach Goal (s)\n(B)', r'Average Acceleration (m/s$^2$)' '\n(C)',
                r'Average Jerk (m/s$^3$)' '\n(D)']

    for i, (end_to_end_datum, waypoint_datum, x_label, e2e_error_bar, waypoint_error_bar) in enumerate(
            zip(end_to_end_data, waypoint_data, x_labels, end_to_end_error_bars, waypoint_error_bars)):
        ax = fig.add_subplot(1, 4, i + 1)

        kwargs_e2e = {}
        kwargs_waypoint = {}

        if i == 0:
            kwargs_e2e['label'] = 'End To End'
            kwargs_waypoint['label'] = waypoint_method_name

        if e2e_error_bar is not None:
            kwargs_e2e['yerr'] = e2e_error_bar

        if waypoint_error_bar is not None:
            kwargs_waypoint['yerr'] = waypoint_error_bar

        rects_waypoint = ax.bar(0, waypoint_datum, bar_width, color=waypoint_color, alpha=waypoint_alpha,
                                **kwargs_waypoint)
        rects_end_to_end = ax.bar(bar_width, end_to_end_datum, bar_width, color=end_to_end_color,
                                  alpha=end_to_end_alpha, **kwargs_e2e)

        # Label Each Rectangle
        rect = rects_end_to_end[0]
        ax.text(rect.get_x() + rect.get_width() / 4., 1.0 * rect.get_height(), '{:.2f}'.format(end_to_end_datum),
                ha='center', va='bottom', fontsize=bar_fontsize)
        rect = rects_waypoint[0]
        ax.text(rect.get_x() + rect.get_width() / 4., 1.0 * rect.get_height(), '{:.2f}'.format(waypoint_datum),
                ha='center', va='bottom', fontsize=bar_fontsize)

        ax.set_xlabel(x_label, fontsize=x_label_fontsize)
        ax.grid(False)

        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks([])

        fig.legend(ncol=2, loc='upper center', prop={'size': legend_size})

    plt.savefig('./plots/bar_chart_v1.pdf', bbox_inches='tight')


def make_bar_chart_v2(x_label_fontsize=12, bar_fontsize=12, legend_size=14, error_bars_mode='std'):
    end_to_end_data, waypoint_data = get_bar_chart_data()

    end_to_end_percentages, end_to_end_common_episode_lengths, end_to_end_common_avg_accelerations, end_to_end_common_avg_jerks = end_to_end_data
    waypoint_percentages, waypoint_success_goals, waypoint_common_episode_lengths, waypoint_common_avg_accelerations, waypoint_common_avg_jerks = waypoint_data

    # Plot Stuff
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)

    end_to_end_data = [end_to_end_percentages['Success'] * 100.0, np.mean(end_to_end_common_episode_lengths),
                       np.mean(end_to_end_common_avg_accelerations), np.mean(end_to_end_common_avg_jerks)]

    waypoint_data = [waypoint_percentages['Success'] * 100.0, np.mean(waypoint_common_episode_lengths),
                     np.mean(waypoint_common_avg_accelerations), np.mean(waypoint_common_avg_jerks)]

    if error_bars_mode == 'std':
        end_to_end_error_bars = [None, np.std(end_to_end_common_episode_lengths),
                                 np.std(end_to_end_common_avg_accelerations), np.std(end_to_end_common_avg_jerks)]
        waypoint_error_bars = [None, np.std(waypoint_common_episode_lengths),
                               np.std(waypoint_common_avg_accelerations), np.std(waypoint_common_avg_jerks)]
    elif error_bars_mode == 'iqr':
        end_to_end_error_bars = [None, np.std(end_to_end_common_episode_lengths),
                                 np.std(end_to_end_common_avg_accelerations), np.std(end_to_end_common_avg_jerks)]
        waypoint_error_bars = [None, np.std(waypoint_common_episode_lengths),
                               np.std(waypoint_common_avg_accelerations), np.std(waypoint_common_avg_jerks)]
    else:
        assert False

    end_to_end_data = np.array(end_to_end_data)
    waypoint_data = np.array(waypoint_data)

    x_labels = ['Success (%)', 'Average Time to Reach Goal (s)', r'Average Acceleration (m/s$^2$)',
                r'Average Jerk (m/s$^3$)']

    for i, (end_to_end_datum, waypoint_datum, x_label, e2e_error_bar, waypoint_error_bar) in enumerate(
            zip(end_to_end_data, waypoint_data, x_labels, end_to_end_error_bars, waypoint_error_bars)):

        ax.clear()

        kwargs_e2e = {}
        kwargs_waypoint = {}

        if i == 0:
            kwargs_e2e['label'] = 'End To End'
            kwargs_waypoint['label'] = waypoint_method_name

        if e2e_error_bar is not None:
            kwargs_e2e['xerr'] = e2e_error_bar

        if waypoint_error_bar is not None:
            kwargs_waypoint['xerr'] = waypoint_error_bar

        # bar_left
        bar_width = .1
        bar_spacing = .12
        rects_waypoint = ax.barh(bar_spacing, waypoint_datum, height=bar_width, color=waypoint_color,
                                 alpha=waypoint_alpha, **kwargs_waypoint)
        rects_end_to_end = ax.barh(0, end_to_end_datum, height=bar_width, color=end_to_end_color,
                                   alpha=end_to_end_alpha, **kwargs_e2e)

        # Label Each Rectangle
        rect = rects_end_to_end[0]
        ax.text(rect.get_x() + rect.get_width() * 1.01, rect.get_y() * 1.15, '{:.2f}'.format(end_to_end_datum),
                ha='left', va='bottom', fontsize=bar_fontsize)
        rect = rects_waypoint[0]
        ax.text(rect.get_x() + rect.get_width() * 1.01, rect.get_y() * .85, '{:.2f}'.format(waypoint_datum),
                ha='left', va='bottom', fontsize=bar_fontsize)

        # ax.set_xlabel(x_label, fontsize=x_label_fontsize)
        ax.grid(False)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_yticks([])
        ax.locator_params(axis='x', nbins=4)

        if i == 0:
            fig.legend(ncol=2, loc='upper center', prop={'size': legend_size})

        fig.savefig('./plots/bar_chart_{:d}_v2.png'.format(i), bbox_inches='tight', pad_inches=0, dpi=400)


def make_bar_chart_v3(x_label_fontsize=12, bar_fontsize=12,
                      legend_size=14, error_bars_mode='std'):
    """
    Same as V2 but plots expert data as well
    """

    end_to_end_data, waypoint_data, expert_data = get_bar_chart_data(return_expert_data=True)

    end_to_end_percentages, end_to_end_common_episode_lengths, end_to_end_common_avg_accelerations, end_to_end_common_avg_jerks = end_to_end_data
    waypoint_percentages, waypoint_success_goals, waypoint_common_episode_lengths, waypoint_common_avg_accelerations, waypoint_common_avg_jerks = waypoint_data
    expert_percentages, expert_common_episode_lengths, expert_common_avg_accelerations, expert_common_avg_jerks = expert_data

    # Plot Stuff
    fig = plt.figure(figsize=(6, 2))
    ax = fig.add_subplot(111)

    end_to_end_data = [end_to_end_percentages['Success'] * 100.0, np.mean(end_to_end_common_episode_lengths),
                       np.mean(end_to_end_common_avg_accelerations), np.mean(end_to_end_common_avg_jerks)]

    waypoint_data = [waypoint_percentages['Success'] * 100.0, np.mean(waypoint_common_episode_lengths),
                     np.mean(waypoint_common_avg_accelerations), np.mean(waypoint_common_avg_jerks)]

    expert_data = [expert_percentages['Success'] * 100., np.mean(expert_common_episode_lengths),
                   np.mean(expert_common_avg_accelerations), np.mean(expert_common_avg_jerks)]

    end_to_end_error_bars = [None, np.std(end_to_end_common_episode_lengths),
                             np.std(end_to_end_common_avg_accelerations), np.std(end_to_end_common_avg_jerks)]
    waypoint_error_bars = [None, np.std(waypoint_common_episode_lengths),
                           np.std(waypoint_common_avg_accelerations), np.std(waypoint_common_avg_jerks)]
    expert_error_bars = [None, np.std(expert_common_episode_lengths),
                         np.std(expert_common_avg_accelerations), np.std(expert_common_avg_jerks)]

    end_to_end_data = np.array(end_to_end_data)
    waypoint_data = np.array(waypoint_data)
    expert_data = np.array(expert_data)

    x_labels = ['Success (%)', 'Average Time to Reach Goal (s)', r'Average Acceleration (m/s$^2$)',
                r'Average Jerk (m/s$^3$)']

    zipped_data = zip(end_to_end_data, waypoint_data, expert_data, x_labels, end_to_end_error_bars, waypoint_error_bars,
                      expert_error_bars)
    for i, (end_to_end_datum, waypoint_datum, expert_datum, x_label, e2e_error_bar, waypoint_error_bar,
            expert_error_bar) in enumerate(zipped_data):

        ax.clear()

        kwargs_e2e = {}
        kwargs_waypoint = {}
        kwargs_expert = {}

        if i == 0:
            kwargs_e2e['label'] = 'End To End'
            kwargs_waypoint['label'] = waypoint_method_name
            kwargs_expert['label'] = 'Expert'

        if e2e_error_bar is not None:
            kwargs_e2e['xerr'] = e2e_error_bar

        if waypoint_error_bar is not None:
            kwargs_waypoint['xerr'] = waypoint_error_bar

        if expert_error_bar is not None:
            kwargs_expert['xerr'] = expert_error_bar

        # bar_left
        bar_width = .1
        bar_spacing = .12
        e2e_start = bar_spacing + bar_width
        expert_start = 2 * bar_spacing + bar_width
        rects_waypoint = ax.barh(3 * bar_spacing, waypoint_datum, height=bar_width, color=waypoint_color,
                                 alpha=waypoint_alpha, **kwargs_waypoint)
        rects_end_to_end = ax.barh(2 * bar_spacing, end_to_end_datum, height=bar_width, color=end_to_end_color,
                                   alpha=end_to_end_alpha, **kwargs_e2e)
        rects_expert = ax.barh(bar_spacing, expert_datum, height=bar_width, color=expert_color, alpha=expert_alpha,
                               **kwargs_expert)

        # Label Each Rectangle
        rect = rects_end_to_end[0]
        ax.text(rect.get_x() + rect.get_width() * 1.01, rect.get_y() * 1.15, '{:.2f}'.format(end_to_end_datum),
                ha='left', va='bottom', fontsize=bar_fontsize)
        rect = rects_waypoint[0]
        ax.text(rect.get_x() + rect.get_width() * 1.01, rect.get_y() * .85, '{:.2f}'.format(waypoint_datum),
                ha='left', va='bottom', fontsize=bar_fontsize)

        rect = rects_expert[0]
        ax.text(rect.get_x() + rect.get_width() * 1.01, rect.get_y() * .85, '{:.2f}'.format(expert_datum),
                ha='left', va='bottom', fontsize=bar_fontsize)

        # ax.set_xlabel(x_label, fontsize=x_label_fontsize)
        ax.grid(False)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_yticks([])
        ax.locator_params(axis='x', nbins=4)

        # if i == 0:
        #     fig.legend(ncol=3, loc='upper center', prop={'size': legend_size})

        fig.savefig('./plots/bar_chart_{:d}_v2.png'.format(i), bbox_inches='tight', pad_inches=0, dpi=400)


def report_numbers():
    """
    Report several metrics for all the planners.
    """

    end_to_end_data, waypoint_data, mapper_data, reactive_data, expert_data = get_bar_chart_data_v2(
        return_expert_data=True)

    end_to_end_percentages, end_to_end_common_episode_lengths, end_to_end_common_avg_accelerations, end_to_end_common_avg_jerks = end_to_end_data
    waypoint_percentages, waypoint_success_goals, waypoint_common_episode_lengths, waypoint_common_avg_accelerations, waypoint_common_avg_jerks = waypoint_data
    mapper_percentages, mapper_success_goals, mapper_common_episode_lengths, mapper_common_avg_accelerations, mapper_common_avg_jerks = mapper_data
    reactive_percentages, reactive_success_goals, reactive_common_episode_lengths, reactive_common_avg_accelerations, reactive_common_avg_jerks = reactive_data
    expert_percentages, expert_common_episode_lengths, expert_common_avg_accelerations, expert_common_avg_jerks = expert_data

    # Success rates
    print('Success rates')
    print('method: expert', expert_percentages['Success'] * 100.0)
    print('method: waypoint', waypoint_percentages['Success'] * 100.0)
    print('method: E2E', end_to_end_percentages['Success'] * 100.0)
    print('method: map', mapper_percentages['Success'] * 100.0)
    print('method: reactive', reactive_percentages['Success'] * 100.0)

    # Time
    print('Time to reach')
    print('method: expert', 'mean', np.mean(expert_common_episode_lengths), 'std',
          np.std(expert_common_episode_lengths))
    print('method: waypoint', 'mean', np.mean(waypoint_common_episode_lengths), 'std',
          np.std(waypoint_common_episode_lengths))
    print('method: E2E', 'mean', np.mean(end_to_end_common_episode_lengths), 'std',
          np.std(end_to_end_common_episode_lengths))
    print('method: map', 'mean', np.mean(mapper_common_episode_lengths), 'std', np.std(mapper_common_episode_lengths))
    print('method: reactive', 'mean', np.mean(reactive_common_episode_lengths), 'std',
          np.std(reactive_common_episode_lengths))

    # Acceleration
    print('Acceleration')
    print('method: expert', 'mean', np.mean(expert_common_avg_accelerations), 'std',
          np.std(expert_common_avg_accelerations))
    print('method: waypoint', 'mean', np.mean(waypoint_common_avg_accelerations), 'std',
          np.std(waypoint_common_avg_accelerations))
    print('method: E2E', 'mean', np.mean(end_to_end_common_avg_accelerations), 'std',
          np.std(end_to_end_common_avg_accelerations))
    print('method: map', 'mean', np.mean(mapper_common_avg_accelerations), 'std',
          np.std(mapper_common_avg_accelerations))
    print('method: reactive', 'mean', np.mean(reactive_common_avg_accelerations), 'std',
          np.std(reactive_common_avg_accelerations))

    # Jerk
    print('Jerk')
    print('method: expert', 'mean', np.mean(expert_common_avg_jerks), 'std', np.std(expert_common_avg_jerks))
    print('method: waypoint', 'mean', np.mean(waypoint_common_avg_jerks), 'std', np.std(waypoint_common_avg_jerks))
    print('method: E2E', 'mean', np.mean(end_to_end_common_avg_jerks), 'std', np.std(end_to_end_common_avg_jerks))
    print('method: map', 'mean', np.mean(mapper_common_avg_jerks), 'std', np.std(mapper_common_avg_jerks))
    print('method: reactive', 'mean', np.mean(reactive_common_avg_jerks), 'std', np.std(reactive_common_avg_jerks))


if __name__ == '__main__':
    # make_bar_chart_v0()
    # make_bar_chart_v1()
    # make_bar_chart_v2(x_label_fontsize=11, bar_fontsize=11)
    # make_bar_chart_v3(x_label_fontsize=11, bar_fontsize=11)
    report_numbers()
