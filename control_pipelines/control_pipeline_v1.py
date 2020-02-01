from utils import utils
import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from optCtrl.lqr import LQRSolver
from trajectory.trajectory import Trajectory, SystemConfig
from control_pipelines.base import ControlPipelineBase
from control_pipelines.control_pipeline_v0 import ControlPipelineV0
from control_pipelines.control_pipeline_v0_helper import ControlPipelineV0Helper


class ControlPipelineV1(ControlPipelineV0):
    """
    This version of control pipeline doesn't squeeze the spline trajectory to the maximum speed of 0.6, and thus doesn't rebin
    """
    pipeline = None

    def __init__(self, params):
        super(ControlPipelineV1, self).__init__(params)

    def _data_file_name(self, file_format='.pkl', v0=None, incorrectly_binned=True):
        """Returns the unique file name given either a starting velocity or incorrectly binned=True."""
        # One of these must be True
        assert (v0 is not None or incorrectly_binned)

        p = self.params
        base_dir = os.path.join(p.dir, 'control_pipeline_v1')
        base_dir = os.path.join(base_dir, 'planning_horizon_{:d}_dt_{:.2f}'.format(
            p.planning_horizon, p.system_dynamics_params.dt))

        base_dir = os.path.join(base_dir, self.system_dynamics.name)
        base_dir = os.path.join(base_dir, self.waypoint_grid.descriptor_string)
        base_dir = os.path.join(base_dir,
                                '{:d}_velocity_bins'.format(p.binning_parameters.num_bins))

        # If using python 2.7 on the real robot the control pipeline will need to be converted to a python 2.7
        # friendly pickle format and will be stored in the subfolder py27.
        if sys.version_info[0] == 2:  # If using python 2.7 on real robot
            base_dir = os.path.join(base_dir, 'py27')

        utils.mkdir_if_missing(base_dir)

        if v0 is not None:
            filename = 'velocity_{:.3f}{:s}'.format(v0, file_format)
        elif incorrectly_binned:
            filename = 'incorrectly_binned{:s}'.format(file_format)
        else:
            assert (False)
        filename = os.path.join(base_dir, filename)
        return filename

    def generate_control_pipeline(self, params=None):
        p = self.params
        # Initialize spline, cost function, lqr solver
        waypoints_egocentric = self._sample_egocentric_waypoints(vf=0.)
        self._init_pipeline()
        pipeline_data = self.helper.empty_data_dictionary()

        with tf.name_scope('generate_control_pipeline'):
            for v0 in self.start_velocities:
                if p.verbose:
                    print('Initial Bin: v0={:.3f}'.format(v0))
                start_config = self.system_dynamics.init_egocentric_robot_config(dt=p.system_dynamics_params.dt,
                                                                                 n=self.waypoint_grid.n, v=v0)
                goal_config = SystemConfig.copy(waypoints_egocentric)
                start_config, goal_config, horizons_n1 = self._dynamically_fit_spline(start_config, goal_config)
                lqr_trajectory, K_nkfd, k_nkf1 = self._lqr(start_config)
                # TODO: Put the initial bin information in here too. This will make debugging much easier.
                data_bin = {'start_configs': start_config,
                            'waypt_configs': goal_config,
                            'start_speeds': self.spline_trajectory.speed_nk1()[:, 0],
                            'spline_trajectories': Trajectory.copy(self.spline_trajectory),
                            'horizons': horizons_n1,
                            'lqr_trajectories': lqr_trajectory,
                            'K_nkfd': K_nkfd,
                            'k_nkf1': k_nkf1}
                self.helper.append_data_bin_to_pipeline_data(pipeline_data, data_bin)
            # This data is incorrectly binned by velocity so collapse it all into one bin before saving it.
            # pipeline_data = self.helper.concat_data_across_binning_dim(pipeline_data)

            self._set_instance_variables(pipeline_data)

        for i, v0 in enumerate(self.start_velocities):
            filename = self._data_file_name(v0=v0)
            data_bin = self.helper.prepare_data_for_saving(pipeline_data, i)
            self.save_control_pipeline(data_bin, filename)

    def _dynamically_fit_spline(self, start_config, goal_config):
        """Fit a spline between start_config and goal_config only keeping points that are dynamically feasible within
        the planning horizon."""
        p = self.params
        times_nk = tf.tile(tf.linspace(0., p.planning_horizon_s, p.planning_horizon)[None], [self.waypoint_grid.n,
                                                                                             1])  # number of waypoints * number of planning horizon. maximum time = 6
        final_times_n1 = tf.ones((self.waypoint_grid.n, 1), dtype=tf.float32) * p.planning_horizon_s
        self.spline_trajectory.fit(start_config, goal_config, final_times_n1=final_times_n1)
        self.spline_trajectory.eval_spline(times_nk, calculate_speeds=True)
        # self.spline_trajectory.rescale_spline_horizon_to_dynamically_feasible_horizon(
        #     speed_max_system=self.system_dynamics.v_bounds[1],
        #     angular_speed_max_system=self.system_dynamics.w_bounds[1], minimum_horizon=p.minimum_spline_horizon)

        # valid_idxs = self.spline_trajectory.find_trajectories_within_a_horizon(p.planning_horizon_s)

        # Valid horizon for each trajectory in the batch
        # in discrete time steps
        self.spline_trajectory.valid_horizons_n1 = tf.ceil(self.spline_trajectory.final_times_n1 / self.spline_trajectory.dt)
        valid_idxs = self.spline_trajectory.check_dynamic_feasibility(speed_max_system=self.system_dynamics.v_bounds[1],
                                                                      angular_speed_max_system=self.system_dynamics.w_bounds[1],
                                                                      horizon_s=p.planning_horizon_s)
        horizons_n1 = self.spline_trajectory.final_times_n1

        # Only keep the valid problems and corresponding splines and horizons
        start_config.gather_across_batch_dim(valid_idxs)
        goal_config.gather_across_batch_dim(valid_idxs)
        horizons_n1 = tf.gather(horizons_n1, valid_idxs)
        self.spline_trajectory.gather_across_batch_dim(valid_idxs)
        return start_config, goal_config, horizons_n1
