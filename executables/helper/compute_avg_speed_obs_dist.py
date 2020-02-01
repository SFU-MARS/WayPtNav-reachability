import tensorflow as tf
import numpy as np
import argparse
import importlib
import pickle
import os

from utils import utils, log_utils

tf.enable_eager_execution(**utils.tf_session_config())

num_files = 500

# Reachability NN
file_folder = '/media/anjianl/My Book/WayPtNav/paper_results/session_2019-10-02_19-58-42_lr-4_reg-6_ckpt19_v2_ctrlhorizon_0.25_500maps/rgb_resnet50_nn_waypoint_simulator/trajectories_with_obsdist'
# Oldcost NN
# file_folder = '/media/anjianl/My Book/WayPtNav/paper_results/session_2019-10-01_20-21-37_pretrained_ctrlhorizon_0.25_500maps/rgb_resnet50_nn_waypoint_simulator/trajectories_with_obsdist'


class ComputeAvgSpeed():
    """
    Compute average speed with respect to distance to the obstacles
    """

    def run(self):

        # Set file path
        testing_dir = os.path.join(file_folder)
        if not os.path.exists(testing_dir):
            raise Exception("no testing folder!")

        # Initialize speed
        # Level 1: <0.15
        speed_level_1 = 0
        speed_num_1 = 0
        # Level 2: 0.15-0.2
        speed_level_2 = 0
        speed_num_2 = 0
        # Level 3: 0.2-0.25
        speed_level_3 = 0
        speed_num_3 = 0
        # Level 4: 0.25-0.3
        speed_level_4 = 0
        speed_num_4 = 0
        # Level 5: 0.3-0.4
        speed_level_5 = 0
        speed_num_5 = 0
        # Level 6: >0.4
        speed_level_6 = 0
        speed_num_6 = 0

        # Main loop
        for j in range(num_files):
            # Find the filename for trajectory data in testing
            filename = os.path.join(testing_dir, 'traj_%i.pkl' % j)

            if not os.path.exists(filename):
                continue

            if j % 50 == 1:
                print("processing episode", j)

            # Load the file
            with open(filename, 'rb') as handle:
                data = pickle.load(handle)

            a = 1

            speed_nk = np.squeeze(data['vehicle_trajectory']['speed_nk1'], axis=2)
            obs_dist_nk = data['distance_to_obstacles_nk']

            # Traverse every trajectory step
            for k in range(speed_nk.shape[1]):
                if obs_dist_nk[0, k] < 0.15:
                    speed_level_1 += speed_nk[0, k]
                    speed_num_1 += 1
                elif 0.15 <= obs_dist_nk[0, k] < 0.2:
                    speed_level_2 += speed_nk[0, k]
                    speed_num_2 += 1
                elif 0.2 <= obs_dist_nk[0, k] < 0.25:
                    speed_level_3 += speed_nk[0, k]
                    speed_num_3 += 1
                elif 0.25 <= obs_dist_nk[0, k] < 0.3:
                    speed_level_4 += speed_nk[0, k]
                    speed_num_4 += 1
                elif 0.3 <= obs_dist_nk[0, k] < 0.4:
                    speed_level_5 += speed_nk[0, k]
                    speed_num_5 += 1
                elif 0.4 <= obs_dist_nk[0, k]:
                    speed_level_6 += speed_nk[0, k]
                    speed_num_6 += 1
                else:
                    print("Wrong with obs dist level!")
                    return False

        print("The average speed at <0.15 obstacle distance is", speed_level_1 / speed_num_1, "The episode num is", speed_num_1)
        print("The average speed at 0.15-0.2 obstacle distance is", speed_level_2 / speed_num_2, "The episode num is", speed_num_2)
        print("The average speed at 0.2-0.25 obstacle distance is", speed_level_3 / speed_num_3, "The episode num is", speed_num_3)
        print("The average speed at 0.25-0.3 obstacle distance is", speed_level_4 / speed_num_4, "The episode num is", speed_num_4)
        print("The average speed at 0.3-0.4 obstacle distance is", speed_level_5 / speed_num_5, "The episode num is", speed_num_5)
        print("The average speed at >0.4 obstacle distance is", speed_level_6 / speed_num_6, "The episode num is", speed_num_6)


if __name__ == '__main__':
    ComputeAvgSpeed().run()
