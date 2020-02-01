from dotmap import DotMap
from utils import utils
import numpy as np
from params.planner_params import create_params as create_planner_params


def create_params():
    p = DotMap()

    # Load the dependencies
    p.planner_params = create_planner_params()

    p.seed = 10  # seed for the simulator (different than for numpy and tf)
    # p.seed = 12

    # Horizons in seconds
    p.episode_horizon_s = 20.0
    # TODO: when generating training data, we use control_horizon_s=1.5s, when testing, we find 0.25s will have the
    #  best performance for WayPtNav-reachability
    p.control_horizon_s = 1.5  # For data generation
    # p.control_horizon_s = 0.25  # Best performance for WayPtNav testing

    # Whether to log videos taken during trajectories
    p.record_video = False

    # Whether or not to log all trajectory data to pickle
    # files when running this simulator
    p.save_trajectory_data = False

    # Define the Objectives

    # Obstacle Avoidance Objective
    p.avoid_obstacle_objective = DotMap(obstacle_margin0=0.3,
                                        obstacle_margin1=0.5,
                                        power=3,
                                        obstacle_cost=1.0)
    # Angle Distance parameters
    p.goal_angle_objective = DotMap(power=1,
                                    angle_cost=.008)
    # Goal Distance parameters
    p.goal_distance_objective = DotMap(power=2,
                                       goal_cost=.08,
                                       goal_margin=.3)

    # Reach avoid 4d parameters
    p.reach_avoid_4d_objective = DotMap(power=1)

    # Reach avoid 3d parameters
    p.reach_avoid_3d_objective = DotMap(power=1)

    # Avoid 4d parameters
    p.avoid_4d_objective = DotMap(avoid_4d_ttr_scale=30)

    p.objective_fn_params = DotMap(obj_type='valid_mean')
    p.reset_params = DotMap(
                            obstacle_map=DotMap(reset_type='random',
                                                params=DotMap(min_n=4, max_n=7,
                                                              min_r=.3, max_r=.8)),
                            start_config=DotMap(
                                                position=DotMap(
                                                    # There could be different reset types
                                                    # 'random': the position is initialized randomly on the
                                                    # map but at least at a distance of the obstacle margin from the
                                                    # obstacle.
                                                    reset_type='random'
                                                ),
                                                heading=DotMap(
                                                    # 'zero': the heading is initialized to zero.
                                                    # 'random': the heading is initialized randomly within the given
                                                    # bounds.
                                                    reset_type='zero',
                                                    bounds=[-np.pi, np.pi-1e-10]
                                                ),
                                                speed=DotMap(
                                                    # For description of reset types see heading parameters above.
                                                    reset_type='zero',
                                                    bounds=[0., 0.6]
                                                ),
                                                ang_speed=DotMap(
                                                    # For description of reset types see heading parameters above.
                                                    reset_type='zero',
                                                    bounds=[-0.5, 0.5],
                                                    gaussian_params=[0.0, .5]  # [mean, variance]
                                                )
                            ),
                            
                            goal_config=DotMap(
                                                position=DotMap(
                                                    # For description of reset types see position parameters in the
                                                    # start_config above.
                                                    reset_type='random'
                                                )
                            )
    )

    p.goal_cutoff_dist = p.goal_distance_objective.goal_margin
    p.goal_dist_norm = p.goal_distance_objective.power  # Default is l2 norm
    p.episode_termination_reasons = ['Timeout', 'Collision', 'Success']
    p.episode_termination_colors = ['b', 'r', 'g']
    p.waypt_cmap = 'winter'

    p.num_validation_goals = 50
    return p
