import numpy as np
import skfmm
import os
import scipy.io as sio
import matlab.engine
import copy
import tensorflow as tf
import time

from utils.voxel_map_4d_utils import VoxelMap4d


class ReachabilityMap(object):
    """
    Maintain a (or more) reachability map (TTR map), which is initialized in MATLAB and computed in C++.
    Input args are specifications of 2D obstacle grid and goal grid.

    """

    def __init__(self, goal_grid_2d, obstacle_grid_2d, map_origin_2d, map_bdry_2d, dx, start_pos_2d, goal_pos_2d,
                 params):
        """
        Maintain the property for 2d goal grid and 2d obstacle grid

        """
        self.goal_grid_2d = goal_grid_2d
        self.obstacle_grid_2d = obstacle_grid_2d
        self.map_size_2d = [goal_grid_2d.shape[0], goal_grid_2d.shape[1]]
        self.map_origin_2d = map_origin_2d
        self.map_bdry_2d = map_bdry_2d
        self.dx = dx
        self.start_pos_2d = start_pos_2d
        self.goal_pos_2d = goal_pos_2d
        self.p = params

        self._reset_variables(update_reach_avoid_4d=True, update_avoid_4d_whole=True)

    def _reset_variables(self, update_reach_avoid_4d=False, update_avoid_4d_whole=False):
        """
        Based on the state space of 4D dubins car, we first generate 4D goal map and obstacle map (signed distance
        map) used for computing TTR map. Then we initialize a VoxelMap4d for TTR map for interpolation purpose.

        """
        # For the purpose of parallel computation for TTR map, we set up different reachablity data fir and name for
        # each thread
        self.reach_avoid_map_4d_path = self.p.reach_avoid_map_4d_path
        self.tmp_path = self.p.tmp_path
        self.avoid_map_4d_path = self.p.avoid_map_4d_path
        self.avoid_map_4d_name = self.p.avoid_map_4d_name
        self.reach_avoid_map_4d_name = self.p.reach_avoid_map_4d_name % (
            self.start_pos_2d[0], self.start_pos_2d[1], self.goal_pos_2d[0], self.goal_pos_2d[1])

        # Set reach_avoid_map_4d tmp path
        self.reach_avoid_4d_map_tmp_path = self.p.MATLAB_PATH + self.p.reach_avoid_4d_map_tmp_path
        # Set avoid_map_4d tmp path
        self.avoid_4d_map_tmp_path = self.p.MATLAB_PATH + self.p.avoid_4d_map_tmp_path

        # Initialize or update reach_avoid_4d TTR map when goal and start positions are changed
        if update_reach_avoid_4d:
            # If it is precomputed, load precomputed 4d reach avoid ttr
            if os.path.isfile(
                    os.path.join(self.p.MATLAB_PATH + self.reach_avoid_map_4d_path + self.reach_avoid_map_4d_name)):
                ttr_4d_reach_avoid_saved = np.load(
                    os.path.join(self.p.MATLAB_PATH + self.reach_avoid_map_4d_path + self.reach_avoid_map_4d_name),
                    allow_pickle=True)
                goal_map_clipped_2d, obstacle_map_clipped_2d, map_clipped_origin_2d, map_clipped_bdry_2d, clipped_dx = \
                    self._reset_clipped_map()
                self._reset_reach_avoid_4d_map_from_saved(ttr_4d_reach_avoid_saved, map_origin_2d=map_clipped_origin_2d)
            # If it is not precomputed, clip the goal map and compute reach avoid TTR map
            else:
                goal_map_clipped_2d, obstacle_map_clipped_2d, map_clipped_origin_2d, map_clipped_bdry_2d, clipped_dx = \
                    self._reset_clipped_map()
                self._reset_reach_avoid_4d_map(goal_map_2d=goal_map_clipped_2d, obstacle_map_2d=obstacle_map_clipped_2d,
                                               map_origin_2d=map_clipped_origin_2d, map_bdry_2d=map_clipped_bdry_2d,
                                               dx=clipped_dx)
                self._compute_reach_avoid_4d_map_LFsweep()

        # Initialize or update avoid_4d TTR map when goal and start positions are changed
        if update_avoid_4d_whole:
            # If it is precomputed, then load precomputed 4d avoid ttr
            if os.path.isfile(os.path.join(self.p.MATLAB_PATH + self.avoid_map_4d_path + self.avoid_map_4d_name)):
                ttr_4d_avoid_whole = np.load(
                    os.path.join(self.p.MATLAB_PATH + self.avoid_map_4d_path + self.avoid_map_4d_name),
                    allow_pickle=True)
                # clipped the 4d avoid ttr, make it the same area as 4d reach avoid
                self._reset_avoid_4d_map_from_whole(ttr_4d_avoid_whole)
            # If it is not precomputed, use the whole obstacle map and compute avoid TTR map. Save it for future use.
            else:
                obstacle_map_whole_2d, map_whole_origin_2d, map_whole_bdry_2d, whole_dx = \
                    self._reset_whole_obstacle_map()
                self._reset_avoid_4d_map(obstacle_map_2d=obstacle_map_whole_2d,
                                         map_origin_2d=map_whole_origin_2d, map_bdry_2d=map_whole_bdry_2d,
                                         dx=whole_dx)
                self._compute_avoid_4d_map_LFsweep()
                ttr_4d_avoid_whole = np.load(
                    os.path.join(self.p.MATLAB_PATH + self.avoid_map_4d_path + self.avoid_map_4d_name))
                self._reset_avoid_4d_map_from_whole(ttr_4d_avoid_whole)

    def _reset_clipped_map(self):
        """
        Every navigation task only happens on a small part of the original map. Therefore, we clip the map w.r.t each
        navigation goal and start position, and only compute TTR on the clipped map

        :return: a clipped map w.r.t the goal and start pos and its specification
        """

        # Configure parameters
        sample_step_x = self.p.sample_step_x_clipped
        sample_step_y = self.p.sample_step_y_clipped

        # Clip the map based on start and goal position. (lower_x, upper_x, lower_y, upper_y)
        lower_x, upper_x, lower_y, upper_y = self._get_clip_index()
        self.clipped_map_bdry = [lower_x, upper_x, lower_y, upper_y]
        clipped_map_origin_2d = np.asarray([lower_x * self.dx, lower_y * self.dx]).astype(np.float32)
        clipped_map_bdry_2d = np.asarray([upper_x * self.dx, upper_y * self.dx]).astype(np.float32)

        # Every time goal is changed, reset goal_map_3d
        clipped_goal_grid_2d = self.goal_grid_2d[lower_x:(upper_x + 1), lower_y:(upper_y + 1)]
        clipped_mask_goal_grid_2d = self._mask_goal(goal_array=clipped_goal_grid_2d,
                                                    dx=self.dx)  # Mask goal position with goal_cutoff_dist = 0.3

        # Down sample clipped map
        clipped_goal_downsampling_2d = clipped_mask_goal_grid_2d[::sample_step_x,
                                       ::sample_step_y]  # Down sample 2D goal array

        if clipped_goal_downsampling_2d.min() >= 0:
            raise Exception("No goals in downsampling array!")

        clipped_goal_map_2d = skfmm.distance(clipped_goal_downsampling_2d, dx=self.dx * sample_step_x * np.ones(2))

        # Every time goal is changed, reset (re-clip) obstacle_map_3d
        clipped_obstacle_grid_2d = self.obstacle_grid_2d[lower_x:(upper_x + 1), lower_y:(upper_y + 1)]

        # Down sample clipped map
        clipped_obstacle_downsampling_2d = clipped_obstacle_grid_2d[::sample_step_x,
                                           ::sample_step_y]  # Down sample 2D goal array

        clipped_obstacle_map_2d = skfmm.distance(clipped_obstacle_downsampling_2d,
                                                 dx=self.dx * sample_step_x * np.ones(2))

        return clipped_goal_map_2d, clipped_obstacle_map_2d, clipped_map_origin_2d, clipped_map_bdry_2d, self.dx * sample_step_x

    def _reset_whole_obstacle_map(self):
        """
        Initialize the whole obstacle map (signed distance map)

        """

        # Configure parameters
        sample_step_x = self.p.sample_step_x_whole
        sample_step_y = self.p.sample_step_y_whole

        # goal_map_2d and obstacle_map_2d are generated independently
        obstacle_downsampling_2d = self.obstacle_grid_2d[::sample_step_x,
                                   ::sample_step_y]  # Down sample 2D obstacle array
        obstacle_map_2d = skfmm.distance(obstacle_downsampling_2d,
                                         dx=self.dx * sample_step_y * np.ones(2))  # Compute signed distance function

        return obstacle_map_2d, copy.copy(self.map_origin_2d), copy.copy(self.map_bdry_2d), self.dx * sample_step_x

    def _get_clip_index(self):
        """
        Compute the index of a clipped map with extension

        """

        # Configure parameters
        extension = self.p.clip_extension

        map_lower_x = np.maximum(np.subtract(np.minimum(self.goal_pos_2d[0], self.start_pos_2d[0]), extension),
                                 self.map_origin_2d[0]).astype(np.float32)
        map_upper_x = np.minimum(np.add(np.maximum(self.goal_pos_2d[0], self.start_pos_2d[0]), extension),
                                 self.map_bdry_2d[0]).astype(np.float32)
        map_lower_y = np.maximum(np.subtract(np.minimum(self.goal_pos_2d[1], self.start_pos_2d[1]), extension),
                                 self.map_origin_2d[1]).astype(np.float32)
        map_upper_y = np.minimum(np.add(np.maximum(self.goal_pos_2d[1], self.start_pos_2d[1]), extension),
                                 self.map_bdry_2d[1]).astype(np.float32)

        new_map_bdry = np.asarray([map_lower_x, map_upper_x, map_lower_y, map_upper_y])

        # Only use when map_origin_2d is [0, 0] (x and y are the same)
        new_map_index = np.floor((new_map_bdry / self.dx) - self.map_origin_2d[0]).astype(np.int32)

        return new_map_index[0], new_map_index[1], new_map_index[2], new_map_index[3]

    def _reset_reach_avoid_4d_map(self, goal_map_2d, obstacle_map_2d, map_origin_2d, map_bdry_2d, dx):
        """
        Initialize or update the 4D map (in state space) as the initialization for reach aovid TTR map computation.
        Require to save temporary map in .mat file for MATLAB script to read

        """

        goal_map_3d = np.repeat(goal_map_2d[:, :, np.newaxis], self.p.theta_dim, axis=2)
        goal_map_4d = np.repeat(goal_map_3d[:, :, :, np.newaxis], self.p.v_dim, axis=3)

        obstacle_map_3d = np.repeat(obstacle_map_2d[:, :, np.newaxis], self.p.theta_dim, axis=2)
        obstacle_map_4d = np.repeat(obstacle_map_3d[:, :, :, np.newaxis], self.p.v_dim, axis=3)

        # For LFSweep ttr computation
        dtheta = np.pi * 2 / self.p.theta_dim
        dv = self.p.v_dv
        self.dx_4d_matlab = matlab.double([dx, dx, dtheta, dv])

        # Configure grids, in matlab type
        map_size_4d = obstacle_map_4d.shape
        map_origin_4d = np.asarray([map_origin_2d[0], map_origin_2d[1], - np.pi, self.p.v_low]).astype(np.float32)
        self.gMin_4d = matlab.double(
            [[map_origin_2d[0]], [map_origin_2d[1]], [- np.pi], [self.p.v_low]])  # Clipped map
        self.gMax_4d = matlab.double(
            [[map_bdry_2d[0]], [map_bdry_2d[1]], [np.pi - dtheta], [self.p.v_high]])  # Clipped map
        self.gN_4d = matlab.double([map_size_4d[0], map_size_4d[1], map_size_4d[2], map_size_4d[3]])

        # goal_map_LFsweep: goal 0, else 100
        goal_map_4d_LFsweep = copy.copy(goal_map_4d)
        goal_map_4d_LFsweep[goal_map_4d <= 0] = 0
        goal_map_4d_LFsweep[goal_map_4d > 0] = 100
        self.goal_reach_avoid_map_4d_LFsweep_list = goal_map_4d_LFsweep.tolist()
        # obstacle_map_LFsweep: obstacle 1, else 0
        obstacle_map_4d_LFsweep = copy.copy(obstacle_map_4d)
        obstacle_map_4d_LFsweep[obstacle_map_4d <= 0] = 1
        obstacle_map_4d_LFsweep[obstacle_map_4d > 0] = 0
        # TODO: add velocity obstacles. first and last dimension of velocity
        obstacle_map_4d_LFsweep[:, :, :, 0] = 1
        obstacle_map_4d_LFsweep[:, :, :, -1] = 1
        self.obstacle_reach_avoid_map_4d_LFsweep_list = obstacle_map_4d_LFsweep.tolist()

        # Save map
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = curr_dir + '/data_tmp'
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        map_to_save = {}
        map_to_save['goal_reach_avoid_map_4d'] = self.goal_reach_avoid_map_4d_LFsweep_list
        map_to_save['obstacle_reach_avoid_map_4d'] = self.obstacle_reach_avoid_map_4d_LFsweep_list
        sio.savemat(data_dir + self.tmp_path + 'reach_avoid_map_4d.mat', map_to_save)

        # Configure dubins car 4d dynamics
        # self.wMax = wMax
        # self.aMax = aMax
        # self.v_high = v_high
        # self.v_low = v_low
        # self.v_high = v_high
        # self.d_max_xy = dMax_xy
        # self.d_max_theta = dMax_theta

        # Initialize a VoxelMap4d
        # TODO: theta dimension is 1 more than original map in matlab! Because in matlab,
        #  we only compute [-pi, pi - dtheta] in theta dimension, but in voxel map for interpolation, we need [-pi,
        #  pi]. Thus we add one more dimension of pi
        self._init_reach_avoid_4d_map(map_size_4d=np.asarray(map_size_4d) + np.asarray([0, 0, 1, 0]),
                                      map_origin_4d=map_origin_4d,
                                      dx_4d=np.asarray([dx, dx, dtheta, dv]))

    def _reset_reach_avoid_4d_map_from_saved(self, ttr_4d_reach_avoid_saved, map_origin_2d):
        """
        If reach_avoid_4d map is precomputed, load it and initialize VoxelMap4d

        """

        # Configure parameters
        sample_step_x = self.p.sample_step_x_clipped

        map_size_4d = ttr_4d_reach_avoid_saved.shape
        map_origin_4d = np.asarray([map_origin_2d[0], map_origin_2d[1], - np.pi, self.p.v_low]).astype(np.float32)

        dtheta = np.pi * 2 / self.p.theta_dim
        dv = self.p.v_dv
        dx = self.dx * sample_step_x

        self._init_reach_avoid_4d_map(map_size_4d=np.asarray(map_size_4d),
                                      map_origin_4d=map_origin_4d,
                                      dx_4d=np.asarray([dx, dx, dtheta, dv]))

        self.reach_avoid_4d_map.voxel_function_4d = tf.convert_to_tensor(ttr_4d_reach_avoid_saved, dtype=tf.float32)

    def _reset_avoid_4d_map(self, obstacle_map_2d, map_origin_2d, map_bdry_2d, dx):
        """
        Initialize or update the 4D map (in state space) as the initialization for aovid TTR map computation.
        Require to save temporary map in .mat file for MATLAB script to read

        """

        obstacle_map_3d = np.repeat(obstacle_map_2d[:, :, np.newaxis], self.p.theta_dim, axis=2)
        obstacle_map_4d = np.repeat(obstacle_map_3d[:, :, :, np.newaxis], self.p.v_dim, axis=3)

        # For LFSweep ttr computation
        dtheta = np.pi * 2 / self.p.theta_dim
        dv = self.p.v_dv
        self.dx_4d_matlab = matlab.double([dx, dx, dtheta, dv])

        # Configure grids, in matlab type
        map_size_4d = obstacle_map_4d.shape
        map_origin_4d = np.asarray([map_origin_2d[0], map_origin_2d[1], - np.pi, self.p.v_low]).astype(np.float32)
        self.gMin_4d = matlab.double(
            [[map_origin_2d[0]], [map_origin_2d[1]], [- np.pi], [self.p.v_low]])  # Clipped map
        self.gMax_4d = matlab.double(
            [[map_bdry_2d[0]], [map_bdry_2d[1]], [np.pi - dtheta], [self.p.v_high]])  # Clipped map
        self.gN_4d = matlab.double([map_size_4d[0], map_size_4d[1], map_size_4d[2], map_size_4d[3]])

        obstacle_map_4d_LFsweep = copy.copy(obstacle_map_4d)
        obstacle_map_4d_LFsweep[obstacle_map_4d <= 0] = 0
        obstacle_map_4d_LFsweep[obstacle_map_4d > 0] = 100
        # TODO: to add velocity obstacles. we set the first and last dimension of velocity in TTR map to be 100
        obstacle_map_4d_LFsweep[:, :, :, 0] = 100
        obstacle_map_4d_LFsweep[:, :, :, -1] = 100
        self.obstacle_avoid_map_4d_LFsweep_list = obstacle_map_4d_LFsweep.tolist()

        # Save map
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = curr_dir + '/data_tmp'
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        map_to_save = {}
        map_to_save['obstacle_avoid_map_4d'] = self.obstacle_avoid_map_4d_LFsweep_list
        sio.savemat(data_dir + self.tmp_path + '/avoid_map_4d_clipped.mat', map_to_save)

        # Configure dubins car 4d dynamics
        # self.wMax = wMax
        # self.aMax = aMax
        # self.v_high = v_high
        # self.v_low = v_low
        # self.v_high = v_high
        # self.d_max_avoid_xy = dMax_avoid_xy
        # self.d_max_avoid_theta = dMax_avoid_theta

        # Initialize a VoxelMap4d
        # TODO: theta dimension is 1 more than original map in matlab! Because in matlab,
        #  we only compute [-pi, pi - dtheta] in theta dimension, but in voxel map for interpolation, we need [-pi,
        #  pi]. Thus we add one more dimension of pi
        self._init_avoid_4d_map(map_size_4d=np.asarray(map_size_4d) + np.asarray([0, 0, 1, 0]),
                                map_origin_4d=map_origin_4d,
                                dx_4d=np.asarray([dx, dx, dtheta, dv]))

    def _reset_avoid_4d_map_from_whole(self, ttr_4d_whole):
        """
        When avoid TTR map is precomputed, clipped it to fits the navigation goal and start (similar to clipped reach
        avoid map).

        """

        # Configure parameters
        sample_step_x = self.p.sample_step_x_clipped
        sample_step_y = self.p.sample_step_y_clipped

        # Clip the map based on start and goal position. (lower_x, upper_x, lower_y, upper_y)
        lower_x, upper_x, lower_y, upper_y = self._get_clip_index()
        clipped_map_origin_2d = np.asarray([lower_x * self.dx, lower_y * self.dx]).astype(np.float32)
        clipped_map_bdry_2d = np.asarray([upper_x * self.dx, upper_y * self.dx]).astype(np.float32)

        # Clipped the ttr
        # Whole ttr not downsample
        clipped_avoid_4d_ttr = ttr_4d_whole[lower_x:(upper_x + 1), lower_y:(upper_y + 1), :, :]

        # Downsample the ttr
        clipped_avoid_4d_ttr = clipped_avoid_4d_ttr[::sample_step_x, ::sample_step_y, :, :]

        # Configure VoxelMap4d params
        map_size_4d = clipped_avoid_4d_ttr.shape
        map_origin_4d = np.asarray([clipped_map_origin_2d[0], clipped_map_origin_2d[1], - np.pi, self.p.v_low]).astype(
            np.float32)
        dtheta = np.pi * 2 / self.p.theta_dim
        dv = self.p.v_dv
        dx = self.dx * sample_step_x

        # Initialize a VoxelMap4d
        self._init_avoid_4d_map(map_size_4d=np.asarray(map_size_4d),
                                map_origin_4d=map_origin_4d,
                                dx_4d=np.asarray([dx, dx, dtheta, dv]))

        self.avoid_4d_map.voxel_function_4d = tf.convert_to_tensor(clipped_avoid_4d_ttr, dtype=tf.float32)

    def _compute_reach_avoid_4d_map_LFsweep(self):
        """
        Compute reach avoid TTR map. Several "threads" are available

        """

        # Set up matlab engine
        self.eng = matlab.engine.start_matlab()
        self.eng.workspace['MATLAB_PATH'] = self.p.MATLAB_PATH
        self.eng.eval("addpath(genpath(MATLAB_PATH))", nargout=0)

        # After computation, ttr has one more dimension on theta. Because we maintain the TTR map in [-pi, pi] in theta
        # dimension

        ttr_value_reach_avoid_4d = self.eng.mainLF_dubins_car_reach_avoid_4d_dxdy_circular(self.dx_4d_matlab,
                                                                                           self.gN_4d,
                                                                                           self.gMin_4d,
                                                                                           self.gMax_4d,
                                                                                           self.p.aMax,
                                                                                           self.p.wMax,
                                                                                           self.p.dMax_xy,
                                                                                           self.p.dMax_theta,
                                                                                           self.reach_avoid_4d_map_tmp_path)

        self.eng.quit()

        ttr_value_reach_avoid_4d_numpy = np.asarray(ttr_value_reach_avoid_4d)
        np.save(os.path.join(self.p.MATLAB_PATH + self.reach_avoid_map_4d_path + self.reach_avoid_map_4d_name),
                ttr_value_reach_avoid_4d_numpy)

        print("new reach avoid 4d ttr is computed and saved!")

        # Wrap the ttr value function into a voxel function
        self.reach_avoid_4d_map.voxel_function_4d = tf.convert_to_tensor(ttr_value_reach_avoid_4d, dtype=tf.float32)

    def _compute_avoid_4d_map_LFsweep(self):
        """
        Compute avoid TTR map. Several "threads" are available

        """

        # Set up matlab engine
        self.eng = matlab.engine.start_matlab()
        self.eng.workspace['MATLAB_PATH'] = self.p.MATLAB_PATH
        self.eng.eval("addpath(genpath(MATLAB_PATH))", nargout=0)

        # After computation, ttr has one more dimension on theta. Because we maintain the TTR map in [-pi, pi] in theta
        # dimension
        ttr_value_avoid_4d = self.eng.mainLF_dubins_car_avoid_4d_dxdy_circular(self.dx_4d_matlab,
                                                                               self.gN_4d,
                                                                               self.gMin_4d,
                                                                               self.gMax_4d,
                                                                               self.p.aMax,
                                                                               self.p.wMax,
                                                                               self.p.dMax_avoid_xy,
                                                                               self.p.dMax_avoid_theta,
                                                                               self.avoid_4d_map_tmp_path)

        self.ttr_avoid_4d = np.asarray(ttr_value_avoid_4d)

        # Save ttr
        np.save(os.path.join(self.p.MATLAB_PATH + self.avoid_map_4d_path + self.avoid_map_4d_name), self.ttr_avoid_4d)
        print("The whole 4d avoid map is saved!")

        # Wrap the ttr value function into a voxel function
        self.avoid_4d_map.voxel_function_4d = tf.convert_to_tensor(ttr_value_avoid_4d, dtype=tf.float32)

    def change_goal(self, start_position_n2, goal_positions_n2):
        """
        Every time when simulator resets the start and the goal, reset the start and goal for TTR map computation

        """

        goal_array_transpose = self.get_goal_array_transpose(goal_positions_n2, self.map_size_2d, self.map_origin_2d,
                                                             self.dx)
        self.goal_grid_2d = goal_array_transpose.transpose()
        self.start_pos_2d = start_position_n2[0]
        self.goal_pos_2d = goal_positions_n2[0]

        self._reset_variables(update_reach_avoid_4d=True, update_avoid_4d_whole=True)
        # self._compute_reachability_map()

    @staticmethod
    def get_goal_array_transpose(goal_positions_n2, map_size_2d, map_origin_2d, dx):
        """
        Given goal position and map size, generate a transposed goal grid

        """
        goal_array_transpose = np.ones((map_size_2d[1], map_size_2d[0]))
        goal_index_x = np.floor((goal_positions_n2[:, 0] / dx) - map_origin_2d[0]).astype(np.int32)
        goal_index_y = np.floor((goal_positions_n2[:, 1] / dx) - map_origin_2d[1]).astype(np.int32)
        goal_array_transpose[goal_index_y, goal_index_x] = -1.
        return goal_array_transpose

    def _mask_goal(self, goal_array, dx):
        """
        Mask the goal state with goal_cutoff_dist = 0.3, set up in the beginning

        """

        # Configure parameters
        mask_distance = self.p.goal_cutoff_dist

        goal_mask = skfmm.distance(goal_array, dx=dx)
        goal_mask[goal_mask <= mask_distance] = -1
        goal_mask[goal_mask > mask_distance] = 1
        return goal_mask

    def _init_reach_avoid_4d_map(self, map_size_4d, map_origin_4d, dx_4d):
        """
        Initialize 4D voxel map for reach avoid TTR map

        """

        self.reach_avoid_4d_map = VoxelMap4d(dx=dx_4d[0], dy=dx_4d[1], dtheta=dx_4d[2], dv=dx_4d[3],
                                             origin_4=map_origin_4d,
                                             map_size_4=map_size_4d)

    def _init_avoid_4d_map(self, map_size_4d, map_origin_4d, dx_4d):
        """
        Initialize 4D voxel map for avoid TTR map

        """

        self.avoid_4d_map = VoxelMap4d(dx=dx_4d[0], dy=dx_4d[1], dtheta=dx_4d[2], dv=dx_4d[3],
                                       origin_4=map_origin_4d,
                                       map_size_4=map_size_4d)

    @staticmethod
    def parse_params(p):
        """
        Parse the parameters to add some additional helpful parameters.
        """
        return p
