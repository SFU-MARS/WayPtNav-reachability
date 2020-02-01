import tensorflow as tf
import numpy as np
import argparse
import importlib
import pickle
import os

from utils import utils, log_utils

tf.enable_eager_execution(**utils.tf_session_config())

num_files = 500
# Ours
# file_folder_testing = '/home/anjianl/Desktop/project/WayPtNav/reproduce_WayptNavResults/session_2019-10-01_16-38-20_lr-4_reg-6_v2_obsdist_2.5_ctrlhorizon_0.25_500maps/rgb_resnet50_nn_waypoint_simulator/trajectories'
# file_folder_testing = '/home/anjianl/Desktop/project/WayPtNav/reproduce_WayptNavResults/session_2019-10-02_19-58-42_lr-4_reg-6_ckpt19_v2_ctrlhorizon_0.25_500maps/rgb_resnet50_nn_waypoint_simulator/trajectories'
# file_folder_testing = '/home/anjianl/Desktop/project/WayPtNav/reproduce_WayptNavResults/session_2019-10-06_16-35-12_lr-4_reg-6_v2_obsdist_2.5_ctrlhorizon_1.5_500maps/rgb_resnet50_nn_waypoint_simulator/trajectories'

# file_folder_testing = '/home/anjianl/Desktop/project/WayPtNav/reproduce_WayptNavResults/session_2019-10-06_20-33-29_pretrain_ctrlhorizon_1.5_500maps/rgb_resnet50_nn_waypoint_simulator/trajectories'
file_folder_testing = '/media/anjianl/My Book/WayPtNav/paper_results/500_map/session_2019-10-02_19-58-42_lr-4_reg-6_ckpt19_v2_ctrlhorizon_0.25_500maps/rgb_resnet50_nn_waypoint_simulator/trajectories'

# Pretrained
# file_folder_baseline = '/home/anjianl/Desktop/project/WayPtNav/reproduce_WayptNavResults/session_2019-10-01_20-21-37_pretrained_ctrlhorizon_0.25_500maps/rgb_resnet50_nn_waypoint_simulator/trajectories'
file_folder_baseline = '/media/anjianl/My Book/WayPtNav/paper_results/500_map/session_2019-10-01_20-21-37_pretrained_ctrlhorizon_0.25_500maps/rgb_resnet50_nn_waypoint_simulator/trajectories'

file_folder_expert = '/media/anjianl/My Book/WayPtNav/paper_results/500_map/session_2019-09-28_12-30-31_reachability_expert_area1_500maps/expert_simulator/trajectories'

# Filter the minimum distance to obstacles
# Only record the metrics when minimum_obs_dist < threshold for expert
lower_limit = -1
upper_limit = 0.15

class RecordMetricsV0(object):

    def run(self):
        """
        Compute the successful rate for each difficult level of episode. Difficulty level is measured by the minimum
        distance to the obstacles according to expert trajectories on the same task
        """

        # Old and new data directories
        testing_dir = os.path.join(file_folder_testing)
        expert_dir = os.path.join(file_folder_expert)
        baseline_dir = os.path.join(file_folder_baseline)
        if not os.path.exists(testing_dir):
            raise Exception("no testing folder!")
        if not os.path.exists(expert_dir):
            raise Exception("no expert folder!")
        if not os.path.exists(baseline_dir):
            raise Exception("no baseline folder!")

        # Set up number to count
        # v1: <= 0.15
        qualified_num_testing_v1 = 0
        qualified_num_baseline_v1 = 0
        total_num_v1 = 0
        # v2: < 0.15 <= 0.2
        qualified_num_testing_v2 = 0
        qualified_num_baseline_v2 = 0
        total_num_v2 = 0
        # v3: < 0.2 <= 0.25
        qualified_num_testing_v3 = 0
        qualified_num_baseline_v3 = 0
        total_num_v3 = 0
        # v4: < 0.25 <= 0.3
        qualified_num_testing_v4 = 0
        qualified_num_baseline_v4 = 0
        total_num_v4 = 0
        # v5: < 0.3 <= 0.4
        qualified_num_testing_v5 = 0
        qualified_num_baseline_v5 = 0
        total_num_v5 = 0
        # v6: < 0.4
        qualified_num_testing_v6 = 0
        qualified_num_baseline_v6 = 0
        total_num_v6 = 0

        # Loop over all trajectories
        for j in range(num_files):
            # Find the filename
            testing_filename = os.path.join(testing_dir, 'traj_%i.pkl' % j)
            expert_filename = os.path.join(expert_dir, 'traj_%i.pkl' % j)
            baseline_filename = os.path.join(baseline_dir, 'traj_%i.pkl' % j)

            if not os.path.exists(expert_filename):
                continue

            # Load the file
            with open(testing_filename, 'rb') as handle_1:
                testing_data = pickle.load(handle_1)
            with open(expert_filename, 'rb') as handle_2:
                expert_data = pickle.load(handle_2)
            with open(baseline_filename, 'rb') as handle_3:
                baseline_data = pickle.load(handle_3)

            # Compute statistics
            # v1:
            if 0.15 >= expert_data['minimum_distance_to_obstacles_n1'] > -1:
                total_num_v1 = total_num_v1 + 1
                if testing_data['episode_type_string'] == 'Success':
                    qualified_num_testing_v1 = qualified_num_testing_v1 + 1
                if baseline_data['episode_type_string'] == 'Success':
                    qualified_num_baseline_v1 = qualified_num_baseline_v1 + 1
            # v2:
            if 0.2 > expert_data['minimum_distance_to_obstacles_n1'] >= 0.15:
                total_num_v2 = total_num_v2 + 1
                if testing_data['episode_type_string'] == 'Success':
                    qualified_num_testing_v2 = qualified_num_testing_v2 + 1
                if baseline_data['episode_type_string'] == 'Success':
                    qualified_num_baseline_v2 = qualified_num_baseline_v2 + 1
            # v3
            if 0.25 > expert_data['minimum_distance_to_obstacles_n1'] >= 0.2:
                total_num_v3 = total_num_v3 + 1
                if testing_data['episode_type_string'] == 'Success':
                    qualified_num_testing_v3 = qualified_num_testing_v3 + 1
                if baseline_data['episode_type_string'] == 'Success':
                    qualified_num_baseline_v3 = qualified_num_baseline_v3 + 1
            # v4:
            if 0.3 > expert_data['minimum_distance_to_obstacles_n1'] >= 0.25:
                total_num_v4 = total_num_v4 + 1
                if testing_data['episode_type_string'] == 'Success':
                    qualified_num_testing_v4 = qualified_num_testing_v4 + 1
                if baseline_data['episode_type_string'] == 'Success':
                    qualified_num_baseline_v4 = qualified_num_baseline_v4 + 1
            # v5
            if 0.4 > expert_data['minimum_distance_to_obstacles_n1'] >= 0.3:
                total_num_v5 = total_num_v5 + 1
                if testing_data['episode_type_string'] == 'Success':
                    qualified_num_testing_v5 = qualified_num_testing_v5 + 1
                if baseline_data['episode_type_string'] == 'Success':
                    qualified_num_baseline_v5 = qualified_num_baseline_v5 + 1
            # v6:
            if 100 > expert_data['minimum_distance_to_obstacles_n1'] >= 0.4:
                total_num_v6 = total_num_v6 + 1
                if testing_data['episode_type_string'] == 'Success':
                    qualified_num_testing_v6 = qualified_num_testing_v6 + 1
                if baseline_data['episode_type_string'] == 'Success':
                    qualified_num_baseline_v6 = qualified_num_baseline_v6 + 1

            if j % 50 == 1:
                print("process", j, "files")

        print("total num is for < 0.15", total_num_v1)
        print("qualifed num for testing is", qualified_num_testing_v1)
        print("our successful rate is", qualified_num_testing_v1 / total_num_v1)
        print("qualifed num for baseline is", qualified_num_baseline_v1)
        print("baseline successful rate is", qualified_num_baseline_v1 / total_num_v1)
        print("-------------------------------------------------------------")

        print("total num is for 0.15-0.2", total_num_v2)
        print("qualifed num for testing is", qualified_num_testing_v2)
        print("our successful rate is", qualified_num_testing_v2 / total_num_v2)
        print("qualifed num for baseline is", qualified_num_baseline_v2)
        print("baseline successful rate is", qualified_num_baseline_v2 / total_num_v2)
        print("-------------------------------------------------------------")

        print("total num is for 0.2-0.25", total_num_v3)
        print("qualifed num for testing is", qualified_num_testing_v3)
        print("our successful rate is", qualified_num_testing_v3 / total_num_v3)
        print("qualifed num for baseline is", qualified_num_baseline_v3)
        print("baseline successful rate is", qualified_num_baseline_v3 / total_num_v3)
        print("-------------------------------------------------------------")

        print("total num is for 0.25-0.3", total_num_v4)
        print("qualifed num for testing is", qualified_num_testing_v4)
        print("our successful rate is", qualified_num_testing_v4 / total_num_v4)
        print("qualifed num for baseline is", qualified_num_baseline_v4)
        print("baseline successful rate is", qualified_num_baseline_v4 / total_num_v4)
        print("-------------------------------------------------------------")

        print("total num is for 0.3-0.4", total_num_v5)
        print("qualifed num for testing is", qualified_num_testing_v5)
        print("our successful rate is", qualified_num_testing_v5 / total_num_v5)
        print("qualifed num for baseline is", qualified_num_baseline_v5)
        print("baseline successful rate is", qualified_num_baseline_v5 / total_num_v5)
        print("-------------------------------------------------------------")

        print("total num is for > 0.4 ", total_num_v6)
        print("qualifed num for testing is", qualified_num_testing_v6)
        print("our successful rate is", qualified_num_testing_v6 / total_num_v6)
        print("qualifed num for baseline is", qualified_num_baseline_v6)
        print("baseline successful rate is", qualified_num_baseline_v6 / total_num_v6)
        print("-------------------------------------------------------------")


if __name__ == '__main__':
    RecordMetricsV0().run()


