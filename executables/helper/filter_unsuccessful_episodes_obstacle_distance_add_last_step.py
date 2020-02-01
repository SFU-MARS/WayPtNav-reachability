import pickle
import numpy as np
import os

"""
Filter timeout or failure episode based on trajectory file
"""

file_folder = '/home/anjianl/Desktop/project/WayPtNav/data/data_generation_no_dist/area5a/v1_100k_seed10'
num_files = 99
new_file_folder = '/home/anjianl/Desktop/project/WayPtNav/data/data_generation_no_dist/area5a/v1_100k_seed10_success'

# Store metadata
store_metadata = False
metadata_prefix = 'img_data_rgb_1024_1024_3_90.00_90.00_0.01_20.00_0.22_18_10_100_80_-45_1.000'

# Only keep successful goals
filter_timeout_episodes = True

# Only filter out failure episode
filter_failure_episodes = False

# Filter minimum obstacle distance
filter_minimum_obstacle_distance_and_timeout = False
minimum_obstacle_distance_in_use = 0.25

def load_and_correct_pickle_files():
    if store_metadata:
        metadata = {}

    # Old and new data directories
    old_dir = os.path.join(file_folder)
    new_dir = os.path.join(new_file_folder)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    original_data_num = 0
    filtered_data_num = 0

    # Last step keys
    # last_step_data_keys = ['last_step_vehicle_state_nk3', 'last_step_vehicle_controls_nk2',
    #                        'last_step_goal_position_n2', 'last_step_goal_position_ego_n2',
    #                        'last_step_optimal_waypoint_n3', 'last_step_optimal_waypoint_ego_n3',
    #                        'last_step_optimal_control_nk2', 'last_step_data_valid_n', 'last_step_human_state_nk3',
    #                        'last_step_human_controls_nk2', 'last_step_human_goal_position_n2',
    #                        'last_step_human_ego_position_n3']

    last_step_data_keys = ['last_step_vehicle_state_nk3', 'last_step_vehicle_controls_nk2',
                           'last_step_goal_position_n2', 'last_step_goal_position_ego_n2',
                           'last_step_optimal_waypoint_n3', 'last_step_optimal_waypoint_ego_n3',
                           'last_step_optimal_control_nk2', 'last_step_data_valid_n']

    # other_data_keys = ['vehicle_state_nk3', 'vehicle_controls_nk2',
    #                    'goal_position_n2', 'goal_position_ego_n2',
    #                    'optimal_waypoint_n3', 'optimal_waypoint_ego_n3', 'waypoint_horizon_n1',
    #                    'optimal_control_nk2', 'episode_type_string_n1', 'episode_number_n1', 'human_ego_position_n3',
    #                    'human_state_nk3', 'human_controls_nk2', 'human_goal_position_n2',
    #                    'human_episode_type_string_n1']

    other_data_keys = ['vehicle_state_nk3', 'vehicle_controls_nk2',
                       'goal_position_n2', 'goal_position_ego_n2',
                       'optimal_waypoint_n3', 'optimal_waypoint_ego_n3', 'waypoint_horizon_n1',
                       'optimal_control_nk2', 'episode_type_string_n1', 'episode_number_n1']

    # other_data_keys = ['vehicle_state_nk3', 'vehicle_controls_nk2',
    #                    'goal_position_n2', 'goal_position_ego_n2',
    #                    'optimal_waypoint_n3', 'optimal_waypoint_ego_n3', 'waypoint_horizon_n1',
    #                    'optimal_control_nk2', 'episode_type_string_n1', 'episode_number_n1',
    #                    'minimum_distance_to_obstacles_n1']

    for j in range(num_files):

        # Find the filename
        filename = os.path.join(old_dir, 'file%i.pkl' % (j + 1))

        # Load the file
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)

        if filter_timeout_episodes:
            print('Original number of samples for file %i: %i' % (
            j + 1, data['goal_position_n2'].shape[0] + data['last_step_goal_position_n2'].shape[0]))

            original_data_num = original_data_num + data['goal_position_n2'].shape[0] + \
                                data['last_step_goal_position_n2'].shape[0]

            # Find the successful episodes
            indices_for_successful_episodes = np.where(data['episode_type_string_n1'] == 'Success')[0]
            successful_episodes_numbers = np.unique(data['episode_number_n1'][indices_for_successful_episodes])

            # Only keep the data corresponding to the successful episodes in the main fields
            data = keep_the_successful_episodes_data(data, other_data_keys, indices_for_successful_episodes)

            # Only keep the data corresponding to the successful episodes in the last trajectory segment fields
            data = keep_the_successful_episodes_data(data, last_step_data_keys, successful_episodes_numbers)

            # Append the last segment data
            data = append_the_last_segment_data_and_delete_last_step_keys(data, other_data_keys)

            print('Filtered number of samples for file %i: %i' % (j + 1, data['goal_position_n2'].shape[0]))
            filtered_data_num = filtered_data_num + data['goal_position_n2'].shape[0]

            # # Do some assertion checks
            # number_of_successful_episodes = successful_episodes_numbers.shape[0]
            # number_of_timeout_episodes = (data['episode_number_n1'][-1] - data['episode_number_n1'][0] + 1) - \
            #                              number_of_successful_episodes
            # assert data['vehicle_state_nk3'].shape[0] == \
            #        data['episode_number_n1'].shape[0] - number_of_timeout_episodes * (
            #        trajectory_segments_per_episode - 1) + number_of_successful_episodes

        # else:
        #     # Just append the last segment data
        #     data = append_the_last_segment_data_and_delete_last_step_keys(data, other_data_keys)

            # # Do some assertion checks
            # assert data['vehicle_state_nk3'].shape[0] == (data['episode_number_n1'][-1] - data['episode_number_n1'][
            #     0] + 1) + data['episode_number_n1'].shape[0]

        if filter_failure_episodes:
            print('Original number of samples for file %i: %i' % (
            j + 1, data['goal_position_n2'].shape[0] + data['last_step_goal_position_n2'].shape[0]))

            original_data_num = original_data_num + data['goal_position_n2'].shape[0] + \
                                data['last_step_goal_position_n2'].shape[0]

            # Find the successful episodes
            indices_for_successful_episodes = np.where(data['episode_type_string_n1'] != 'Failure')[0]
            successful_episodes_numbers = np.unique(data['episode_number_n1'][indices_for_successful_episodes])

            # Only keep the data corresponding to the successful episodes in the main fields
            data = keep_the_successful_episodes_data(data, other_data_keys, indices_for_successful_episodes)

            # Only keep the data corresponding to the successful episodes in the last trajectory segment fields
            data = keep_the_successful_episodes_data(data, last_step_data_keys, successful_episodes_numbers)

            # Append the last segment data
            data = append_the_last_segment_data_and_delete_last_step_keys(data, other_data_keys)

            print('Filtered number of samples for file %i: %i' % (j + 1, data['goal_position_n2'].shape[0]))
            filtered_data_num = filtered_data_num + data['goal_position_n2'].shape[0]

        # Filter out minimum obstacle distance
        if filter_minimum_obstacle_distance_and_timeout:
            print('Original number of samples for file %i: %i' % (
                j + 1, data['goal_position_n2'].shape[0] + data['last_step_goal_position_n2'].shape[0]))
            original_data_num = original_data_num + data['goal_position_n2'].shape[0] + \
                                data['last_step_goal_position_n2'].shape[0]

            # Find the successful episodes
            indices_for_successful_episodes = np.where((data['minimum_distance_to_obstacles_n1'] >=
                                                       minimum_obstacle_distance_in_use) &
                                                       (data['episode_type_string_n1'] == 'Success'))[0]

            successful_episodes_numbers = np.unique(data['episode_number_n1'][indices_for_successful_episodes])

            # Only keep the data corresponding to the successful episodes in the main fields
            data = keep_the_successful_episodes_data(data, other_data_keys, indices_for_successful_episodes)

            # Only keep the data corresponding to the successful episodes in the last trajectory segment fields
            data = keep_the_successful_episodes_data(data, last_step_data_keys, successful_episodes_numbers)

            # Append the last segment data
            data = append_the_last_segment_data_and_delete_last_step_keys(data, other_data_keys)

            print('Filtered number of samples for file %i: %i' % (j + 1, data['goal_position_n2'].shape[0]))
            filtered_data_num = filtered_data_num + data['goal_position_n2'].shape[0]

        # Save the file
        filename = os.path.join(new_dir, 'file%i.pkl' % (j + 1))
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Store the metadata
        if store_metadata:
            metadata_key = os.path.join(new_dir, metadata_prefix, 'file%i.pkl' % (j + 1))
            metadata[metadata_key] = data['vehicle_state_nk3'].shape[0]

    # Save the metadata file
    if store_metadata:
        filename = os.path.join(new_dir, 'metadata.pkl')
        with open(filename, 'wb') as handle:
            pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("total original data num is", original_data_num)
    print("total remain data num is", filtered_data_num)
    ratio_of_filtered_out = (original_data_num - filtered_data_num) / original_data_num
    print('total filter out', original_data_num - filtered_data_num)
    print('total filterd out ratio', ratio_of_filtered_out)


def keep_the_successful_episodes_data(data, keys_to_operate_on, indices_to_keep):
    for key in keys_to_operate_on:
        data[key] = data[key][indices_to_keep]
    return data


def append_the_last_segment_data_and_delete_last_step_keys(data, data_keys_to_append):
    new_data = {}
    data_keys = data.keys()
    num_episodes = data['last_step_vehicle_state_nk3'].shape[0]
    counter_end = 0
    counter_start = 0
    max_counter = data['episode_number_n1'].shape[0]
    for i in range(num_episodes):
        counter_start = counter_end * 1
        episode_number = data['episode_number_n1'][counter_start]
        while (counter_end < max_counter) and (data['episode_number_n1'][counter_end] == episode_number):
            counter_end = counter_end + 1
        if i == 0:
            for key in data_keys_to_append:
                last_step_key = 'last_step_' + key
                if key in ['episode_type_string_n1']:
                    new_data[key] = np.concatenate((data[key][counter_start:counter_end], np.array(['Success'])),
                                                   axis=0)
                elif key in ['episode_number_n1']:
                    new_data[key] = np.concatenate((data[key][counter_start:counter_end], np.array([episode_number])),
                                                   axis=0)
                elif last_step_key in data_keys:
                    new_data[key] = np.concatenate((data[key][counter_start:counter_end], data[last_step_key][i:i + 1]),
                                                   axis=0)
        else:
            for key in data_keys_to_append:
                last_step_key = 'last_step_' + key
                if key in ['episode_type_string_n1']:
                    new_data[key] = np.concatenate(
                        (new_data[key], data[key][counter_start:counter_end], np.array(['Success'])), axis=0)
                elif key in ['episode_number_n1']:
                    new_data[key] = np.concatenate(
                        (new_data[key], data[key][counter_start:counter_end], np.array([episode_number])), axis=0)
                elif last_step_key in data_keys:
                    new_data[key] = np.concatenate(
                        (new_data[key], data[key][counter_start:counter_end], data[last_step_key][i:i + 1]), axis=0)
    return new_data

    # for key in data_keys_to_append:
    #     last_step_key = 'last_step_' + key
    #     if last_step_key in data_keys:
    #         data[key] = np.concatenate((data[key], data[last_step_key]), axis=0)
    # # # This if-else conditions are implemented because of a bug in the collection of last step data
    # # if key in ['goal_position_n2', 'goal_position_ego_n2']:
    # #     data[key] = np.concatenate((data[key], data[last_step_key][:, 0, 0:2]), axis=0)
    # # elif key in ['optimal_waypoint_n3', 'optimal_waypoint_ego_n3']:
    # #     data[key] = np.concatenate((data[key], data[last_step_key][:, 0]), axis=0)
    # # else:
    # #     data[key] = np.concatenate((data[key], data[last_step_key]), axis=0)
    #         del data[last_step_key]
    # return data


if __name__ == '__main__':
    load_and_correct_pickle_files()
