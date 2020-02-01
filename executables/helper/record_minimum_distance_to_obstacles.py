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
# file_folder = '/media/anjianl/My Book/WayPtNav/paper_results/session_2019-10-02_19-58-42_lr-4_reg-6_ckpt19_v2_ctrlhorizon_0.25_500maps/rgb_resnet50_nn_waypoint_simulator/trajectories'
# new_file_folder = '/media/anjianl/My Book/WayPtNav/paper_results/session_2019-10-02_19-58-42_lr-4_reg-6_ckpt19_v2_ctrlhorizon_0.25_500maps/rgb_resnet50_nn_waypoint_simulator/trajectories_with_obsdist'
# Oldcost NN
file_folder = '/media/anjianl/My Book/WayPtNav/paper_results/session_2019-10-01_20-21-37_pretrained_ctrlhorizon_0.25_500maps/rgb_resnet50_nn_waypoint_simulator/trajectories'
new_file_folder = '/media/anjianl/My Book/WayPtNav/paper_results/session_2019-10-01_20-21-37_pretrained_ctrlhorizon_0.25_500maps/rgb_resnet50_nn_waypoint_simulator/trajectories_with_obsdist'


class RecordMinDistToObs(object):

    def run(self):
        """
        Compute minimum distance to the obstacle along the trajectory. Save it in the traj file
        """

        simulator = self.get_simulator()

        # Old and new data directories
        old_dir = os.path.join(file_folder)
        new_dir = os.path.join(new_file_folder)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        for j in range(num_files):
            # # Find the filename for generated data
            # filename = os.path.join(old_dir, 'file%i.pkl' % (j + 1))
            # Find the filename for trajectory data in testing
            filename = os.path.join(old_dir, 'traj_%i.pkl' % j)

            if not os.path.exists(filename):
                continue

            # Load the file
            with open(filename, 'rb') as handle:
                data = pickle.load(handle)

            if j % 50 == 1:
                print("processing episode", j)

            # # Get trajectory from generated data
            # trajectory = data['vehicle_state_nk3'][:, :, 0:2]
            # Get trajectory from testing data
            trajectory = data['vehicle_trajectory']['position_nk2']

            # Record distance to obstacles for every step in the trajectory
            obstacle_dists_nk = simulator.obstacle_map.dist_to_nearest_obs(trajectory).numpy()
            data['distance_to_obstacles_nk'] = obstacle_dists_nk

            # Record minimum distance to the obstacles for the entire trajectory
            min_obstacle_dists_n1 = np.amin(obstacle_dists_nk, axis=1)
            # print('min obs dist is', min_obstacle_dists_n1)
            data['minimum_distance_to_obstacles_n1'] = min_obstacle_dists_n1

            # Save the file
            filename = os.path.join(new_dir, 'traj_%i.pkl' % (j + 1))
            with open(filename, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_simulator(self):

        parser = argparse.ArgumentParser(description='Process the command line inputs')
        parser.add_argument("-p", "--params", required=True, help='the path to the parameter file')
        args = parser.parse_args()

        p = self.create_params(args.params)

        p.simulator_params = p.data_creation.simulator_params
        p.simulator_params.simulator.parse_params(p.simulator_params)

        simulator = p.simulator_params.simulator(p.simulator_params)

        return simulator

    def create_params(self, param_file):
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


if __name__ == '__main__':
    RecordMinDistToObs().run()