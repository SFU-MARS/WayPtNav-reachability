from obstacles.sbpd_map import SBPDMap
from simulators.simulator import Simulator


class SBPDSimulator(Simulator):
    name = 'SBPD_Simulator'

    def __init__(self, params):
        assert(params.obstacle_map_params.obstacle_map is SBPDMap)
        super(SBPDSimulator, self).__init__(params=params)

    def get_observation(self, config=None, pos_n3=None, **kwargs):
        """
        Return the robot's observation from configuration config
        or pos_nk3.
        """
        return self.obstacle_map.get_observation(config=config, pos_n3=pos_n3, **kwargs)

    def get_observation_from_data_dict_and_model(self, data_dict, model):
        """
        Returns the robot's observation from the data inside data_dict,
        using parameters specified by the model.
        """
        if hasattr(model, 'occupancy_grid_positions_ego_1mk12'):
            kwargs = {'occupancy_grid_positions_ego_1mk12':
                      model.occupancy_grid_positions_ego_1mk12}
        else:
            kwargs = {}

        img_nmkd = self.get_observation(pos_n3=data_dict['vehicle_state_nk3'][:, 0],
                                        **kwargs)
        return img_nmkd

    def _reset_obstacle_map(self, rng):
        """
        For SBPD the obstacle map does not change
        between episodes.
        """
        return False

    def _update_fmm_map(self):
        """
        For SBPD the obstacle map does not change,
        so just update the goal position.
        """
        if hasattr(self, 'fmm_map'):
            goal_pos_n2 = self.goal_config.position_nk2()[:, 0]
            # Change goal, and recompute reach_avoid_map
            self.fmm_map.change_goal(goal_pos_n2)
        else:
            self.fmm_map = self._init_fmm_map() # given goal and obstacle_occupancy_grids, initialize fmm_angle_map, fmm_distance_map and goal_grid_mn

    def _get_reachability_map(self):
        """
        For SBPD, given obstalce map and goal array, initialize (or update) reachability map

        """
        # If there is reachability map, update it with changed goal. Otherwise, init them
        # The obstacle map won't change, because every time generating data, we only use 1 map
        if hasattr(self, 'reachability_map'):
            start_pos_n2 = self.start_config.position_nk2()[:, 0].numpy()
            goal_pos_n2 = self.goal_config.position_nk2()[:, 0].numpy()
            self.reachability_map.change_goal(start_position_n2=start_pos_n2, goal_positions_n2=goal_pos_n2)
        else:
            self.reachability_map = self._init_reachability_map()

    def _init_obstacle_map(self, rng):
        """ Initializes the sbpd map."""
        p = self.params.obstacle_map_params
        return p.obstacle_map(p)

    def _render_obstacle_map(self, ax):
        p = self.params
        self.obstacle_map.render_with_obstacle_margins(ax, start_config=self.start_config,
                                                       margin0=p.avoid_obstacle_objective.obstacle_margin0,
                                                       margin1=p.avoid_obstacle_objective.obstacle_margin1)
