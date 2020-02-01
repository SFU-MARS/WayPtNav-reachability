from dotmap import DotMap
from reachability.reachability_map import ReachabilityMap

import numpy as np


def create_params():
    p = DotMap()

    p.reachability_map = ReachabilityMap

    # Map operation
    # Obstacle map resolution=0.05m. To shorten the computation time, we down sample the map by the following step size
    p.sample_step_x_clipped = 2
    p.sample_step_y_clipped = 2
    p.sample_step_x_whole = 1
    p.sample_step_y_whole = 1
    # In a navigation task, we assume the robot successfully reaches the goal if it is within 0.3m of the goal point.
    # However, in the TTR computation, in order to lead the robot to the goal more accurately, we set a smaller
    # goal_cutoff_dist
    p.goal_cutoff_dist = 0.1
    p.obstacle_augment_dist = 0.15
    # When computing reach-avoid TTR, the whole map is too large. Thus we first clip the map around the start and the
    # goal position of each episode. The clipped map augment the boundary of the start-goal rectangle by clip_extension
    p.clip_extension = np.float32(2.0)

    # System dynamics (state constraints and discretization)
    p.theta_dim = int(30)
    p.v_dim = int(9)
    p.v_dv = float(.1)
    p.wMax = float(1.1)
    p.aMax = float(.4)
    p.v_high = float(.7)
    p.v_low = float(-.1)

    # System dynamics (disturbances constraints)
    # Additive disturbances
    # Disturbances used for modeling prediction errors
    p.dMax_xy = float(0.05)
    p.dMax_theta = float(0.15)
    p.dMax_avoid_xy = float(0.05)
    p.dMax_avoid_theta = float(0.15)
    # # No disturbance version
    # p.dMax_xy = float(0.0)
    # p.dMax_theta = float(0.0)
    # p.dMax_avoid_xy = float(0.0)
    # p.dMax_avoid_theta = float(0.0)

    # Set Matlab path
    p.MATLAB_PATH = '/home/anjianl/Desktop/project/WayPtNav/reachability'

    # Set data path based on threads
    p.thread = 'v1'

    return p


def create_reachability_data_dir_params(p):
    if p.thread == 'v1':
        # Configure reach avoid map 4d
        p.reach_avoid_map_4d_path = '/data_tmp/reach_avoid_map_4d/v1/area3/'
        p.reach_avoid_map_4d_name = 'area3_start_%.2f_%.2f_goal_%.2f_%.2f.npy'
        # Configure avoid map 4d
        p.avoid_map_4d_path = '/data_tmp/avoid_map_4d/v1/'
        p.avoid_map_4d_name = 'ttr_avoid_map_4d_whole_area3_no_dist.npy'
        # Configure map tmp
        p.reach_avoid_4d_map_tmp_path = '/data_tmp/tmp/v1/reach_avoid_map_4d.mat'
        p.avoid_4d_map_tmp_path = '/data_tmp/tmp/v1/avoid_map_4d_clipped.mat'
        p.tmp_path = '/tmp/v1/'

    elif p.thread == 'v2':
        # Configure reach avoid map 4d
        p.reach_avoid_map_4d_path = '/data_tmp/reach_avoid_map_4d/v2/area4/'
        p.reach_avoid_map_4d_name = 'area4_start_%.2f_%.2f_goal_%.2f_%.2f.npy'
        # Configure avoid map 4d
        p.reach_avoid_4d_map_tmp_path = '/data_tmp/tmp/v2/reach_avoid_map_4d.mat'
        p.avoid_map_4d_name = 'ttr_avoid_map_4d_whole_area4_no_dist.npy'
        # Configure map tmp
        p.avoid_4d_map_tmp_path = '/data_tmp/tmp/v2/avoid_map_4d_clipped.mat'
        p.tmp_path = '/tmp/v2/'
        p.avoid_map_4d_path = '/data_tmp/avoid_map_4d/v2/'

    elif p.thread == 'v3':
        # Configure reach avoid map 4d
        p.reach_avoid_map_4d_path = '/data_tmp/reach_avoid_map_4d/v3/area5a/'
        p.reach_avoid_map_4d_name = "area5a_start_%.2f_%.2f_goal_%.2f_%.2f.npy"
        # Configure avoid map 4d
        p.reach_avoid_4d_map_tmp_path = '/data_tmp/tmp/v3/reach_avoid_map_4d.mat'
        p.avoid_map_4d_name = 'ttr_avoid_map_4d_whole_area5a_no_dist.npy'
        # Configure map tmp
        p.avoid_4d_map_tmp_path = '/data_tmp/tmp/v3/avoid_map_4d_clipped.mat'
        p.tmp_path = '/tmp/v3/'
        p.avoid_map_4d_path = '/data_tmp/avoid_map_4d/v3/'
