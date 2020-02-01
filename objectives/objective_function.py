import tensorflow as tf
import numpy as np
import time

# Within the goal_mask distance to the goal, the robot is assumed to reach the goal
goal_mask = 0.3

class Objective(object):
    def evaluate_objective(self, trajectory):
        raise NotImplementedError


from objectives.reach_avoid_4d import ReachAvoid4d
from objectives.reach_avoid_3d import ReachAvoid3d
from objectives.avoid_4d import Avoid4d


from objectives.goal_distance import GoalDistance


class ObjectiveFunction(object):
    """
    Define an objective function.
    """

    def __init__(self, params):
        self.params = params
        self.objectives = []

    def add_objective(self, objective):
        """
        Add an objective to the objective function.
        """
        self.objectives.append(objective)

    def evaluate_function_by_objective(self, trajectory):
        """
        Evaluate each objective corresponding to a system trajectory.
        """
        objective_values_by_tag = []
        reachability_cost = False

        # judge the cost to use, if there is ReachAvoid4d, there must be reachability cost
        for objective in self.objectives:
            if isinstance(objective, ReachAvoid4d):
                reachability_cost = True

        if reachability_cost:
            for objective in self.objectives:
                if isinstance(objective, ReachAvoid4d) or isinstance(objective, Avoid4d) or isinstance(objective, GoalDistance):
                    objective_values_by_tag.append([objective.tag, objective.evaluate_objective(trajectory)])
        else:
            for objective in self.objectives:
                objective_values_by_tag.append([objective.tag, objective.evaluate_objective(trajectory)])


        # for objective in self.objectives:
        #     # if isinstance(objective, ReachAvoid4d) or isinstance(objective, Avoid4d):
        #     # if isinstance(objective, ReachAvoid4d):
        #     objective_values_by_tag.append([objective.tag, objective.evaluate_objective(trajectory)])

        return objective_values_by_tag

    def evaluate_function(self, trajectory):
        """
        Evaluate the entire objective function corresponding to a system trajectory.
        """
        objective_values_by_tag = self.evaluate_function_by_objective(trajectory)
        objective_function_values = 0.
        objective_distance_to_goal = 0.
        objective_function_values_init = 0.

        reachability_cost = False

        # Judge if we are using reachability cost
        for tag, objective_values in objective_values_by_tag:
            if tag == 'reach_avoid_4d':
                reachability_cost = True

        # No freezing cost!
        if reachability_cost:
            for tag, objective_values in objective_values_by_tag:
                if tag == 'reach_avoid_4d' or 'avoid_4d':
                    objective_function_values += self._reduce_objective_values(trajectory, objective_values)
        else:
            for tag, objective_values in objective_values_by_tag:
                objective_function_values += self._reduce_objective_values(trajectory, objective_values)

        # ## Freeze the sum of 2 costs, at the minimum of the sum of cost
        # if reachability_cost:
        #     for tag, objective_values in objective_values_by_tag:
        #         if tag == 'reach_avoid_4d' or 'avoid_4d':
        #             objective_function_values += objective_values
        #         if tag == 'distance_to_goal':
        #             objective_distance_to_goal += objective_values
        #     try:
        #         ## Freeze_v1, freeze at the minimum cost of sum: reach_avoid + avoid
        #         # objective_function_values = self._freeze_cost_v1(objective_function_values, objective_distance_to_goal)
        #
        #         # Freeze v2, freeze at the minimum cost of only reach_avoid
        #         objective_function_values = self._freeze_cost_v2(objective_values_by_tag, objective_function_values, objective_distance_to_goal)
        #         objective_function_values = self._reduce_objective_values(trajectory, objective_function_values)
        #     except ValueError:
        #         print("cannot freeze in total cost")
        #         objective_function_values = self._reduce_objective_values(trajectory, objective_function_values)
        # else:
        #     for tag, objective_values in objective_values_by_tag:
        #         objective_function_values += self._reduce_objective_values(trajectory, objective_values)

        # ## Freeze the sum of 2 costs, at the minimum of the reach_avoid cost
        # if reachability_cost:
        #     for tag, objective_values in objective_values_by_tag:
        #         if tag == 'reach_avoid_4d' or 'avoid_4d':
        #             objective_function_values = self._freeze_cost_v2(objective_values, objective_distance_to_goal)
        #             objective_function_values += self._reduce_objective_values(trajectory, objective_values)
        # else:
        #     for tag, objective_values in objective_values_by_tag:
        #         objective_function_values += self._reduce_objective_values(trajectory, objective_values)

        return objective_function_values

    def _reduce_objective_values(self, trajectory, objective_values):
        """Reduce objective_values according to
        self.params.obj_type."""
        if self.params.obj_type == 'mean':
            res = tf.reduce_mean(objective_values, axis=1)
        elif self.params.obj_type == 'valid_mean':
            valid_mask_nk = trajectory.valid_mask_nk
            obj_sum = tf.reduce_sum(objective_values * valid_mask_nk, axis=1)
            res = obj_sum / trajectory.valid_horizons_n1[:, 0]
        else:
            assert (False)
        return res

    def _freeze_cost_v1(self, objective_values, objective_distance_to_goal):

        obj_val_np = objective_values.numpy()
        goal_dist_np = objective_distance_to_goal.numpy()

        size_val = obj_val_np.shape

        if np.min(goal_dist_np) <= goal_mask:
            for i in range(size_val[0]):
                if np.min(goal_dist_np[i, :]) < goal_mask:
                    goal_enter_index = np.min(np.where(goal_dist_np[i, :] < goal_mask))
                    if goal_enter_index < size_val[0]:
                        min_cost = np.min(obj_val_np[i, goal_enter_index:size_val[0]])
                        min_cost_index = np.argmin(obj_val_np[i, goal_enter_index:size_val[0]])
                        obj_val_np[i, (goal_enter_index+min_cost_index):] = min_cost

        return tf.constant(obj_val_np, dtype=tf.float32)

    def _freeze_cost_v2(self, objective_values_by_tag, objective_values, objective_distance_to_goal):

        obj_val_np = objective_values.numpy()
        goal_dist_np = objective_distance_to_goal.numpy()

        size_val = obj_val_np.shape

        # First, find the index to start freeze from only reach_avoid_cost
        for tag, objective_values in objective_values_by_tag:
            if tag == 'reach_avoid_4d':
                reach_avoid_4d_values = objective_values

        # # Freeze the cost when it collides with the obstacles
        # for i in range(size_val[0]):
        #     if np.max(obj_val_np[i, :]) >= 300:
        #         obstacle_enter_index = np.min(np.where(obj_val_np[i, :] >= 300))
        #         if obstacle_enter_index < size_val[0]:
        #             # Freeze the cost when hitting obstacles
        #             max_cost = np.max(obj_val_np[i, obstacle_enter_index:size_val[0]])
        #             obj_val_np[i, obstacle_enter_index:] = max_cost

        # Freeze the cost to previous cost when it reach the goal
        if np.min(goal_dist_np) <= goal_mask:
            for i in range(size_val[0]):
                if np.min(goal_dist_np[i, :]) < goal_mask:
                    goal_enter_index = np.min(np.where(goal_dist_np[i, :] < goal_mask))
                    if goal_enter_index < size_val[0] and np.max(reach_avoid_4d_values[i, :goal_enter_index + 1]) < 100:
                        # set all cost to be reach aovid cost
                        # min_cost = np.min(reach_avoid_4d_values[i, goal_enter_index:size_val[0]])
                        # Set all cost to be the previous cost
                        min_cost = np.min(obj_val_np[i, goal_enter_index:size_val[0]])
                        min_cost_index = np.argmin(reach_avoid_4d_values[i, goal_enter_index:size_val[0]])
                        obj_val_np[i, (goal_enter_index+min_cost_index):] = min_cost

        # # Freeze the cost to only reach_avoid cost when it reach the goal
        # if np.min(goal_dist_np) <= goal_mask:
        #     for i in range(size_val[0]):
        #         if np.min(goal_dist_np[i, :]) < goal_mask:
        #             goal_enter_index = np.min(np.where(goal_dist_np[i, :] < goal_mask))
        #             if goal_enter_index < size_val[0] and np.max(reach_avoid_4d_values[i, :goal_enter_index + 1]) < 100:
        #                 # set all cost to be reach aovid cost
        #                 # min_cost = np.min(reach_avoid_4d_values[i, goal_enter_index:size_val[0]])
        #                 # Set all cost to be the previous cost
        #                 min_cost = np.min(reach_avoid_4d_values[i, goal_enter_index:size_val[0]])
        #                 min_cost_index = np.argmin(reach_avoid_4d_values[i, goal_enter_index:size_val[0]])
        #                 obj_val_np[i, (goal_enter_index+min_cost_index):] = min_cost

        # if np.min(goal_dist_np) <= goal_mask:
        #     for i in range(size_val[0]):
        #         if np.min(goal_dist_np[i, :]) < goal_mask and np.max(reach_avoid_4d_values[i, :goal_enter_index + 1]) < 100:
        #             goal_enter_index = np.min(np.where(goal_dist_np[i, :] < goal_mask))
        #             if goal_enter_index < size_val[0]:
        #                 min_cost = np.min(obj_val_np[i, goal_enter_index:size_val[0]])
        #                 min_cost_index = np.argmin(obj_val_np[i, goal_enter_index:size_val[0]])
        #                 obj_val_np[i, (goal_enter_index+min_cost_index):] = min_cost

        return tf.constant(obj_val_np, dtype=tf.float32)

    def _freeze_cost(self, objective_values):

        obj_val_np = objective_values.numpy()

        size_val = obj_val_np.shape

        # Freeze the cost when get to the goal
        for i in range(size_val[0]):
            min_val = np.amin(obj_val_np[i, :])
            min_index = np.argmin(obj_val_np[i, :])

            if min_val <= 0 and min_index < size_val[1]:
                obj_val_np[i, min_index:] = 0

        return tf.constant(obj_val_np, dtype=tf.float32)