import tensorflow as tf
import numpy as np
from objectives.objective_function import Objective


class ReachAvoid4d(Objective):
    """
    Reach avoid cost measured on 4D reach-avoid TTC value map
    """

    def __init__(self, params, reachability_map):
        self.p = params
        self.reachability_map = reachability_map
        self.tag = 'reach_avoid_4d'

    def compute_reach_avoid_4d(self, trajectory):

        return self.reachability_map.reach_avoid_4d_map.compute_voxel_function(trajectory.position_nk2(),
                                                                               trajectory.heading_nk1(),
                                                                               trajectory.speed_nk1())

    def evaluate_objective(self, trajectory):

        # the value represents the time to reach the goal, 100 indicates in the obstacles
        reach_avoid_4d = self.compute_reach_avoid_4d(trajectory)

        # # If freezing the cost when entering the obstacles
        # try:
        #     reach_avoid_4d = self._freeze_cost_obstacle_enter(reach_avoid_4d)
        # except ValueError:
        #     print("cannot freeze_cost_obstacle_enter in reach avoid 4d")

        return reach_avoid_4d

    def _freeze_cost_obstacle_enter(self, objective_values):

        obj_val_np = objective_values.numpy()

        size_val = obj_val_np.shape

        try:
            for i in range(size_val[0]):
                min_val = np.amin(obj_val_np[i, :])
                min_index = np.argmin(obj_val_np[i, :])
                max_val = np.amax(obj_val_np[i, :])
                max_index = np.argmax(obj_val_np[i, :])

                # If max pos first
                if max_index < min_index:
                    # First check whether to freeze obstacle cost
                    if max_val >= 100 and max_index < size_val[1]:
                        obj_val_np[i, max_index:] = 100
                    # elif min_val == 0 and min_index < size_val[1]:
                    #     obj_val_np[i, min_index:] = 0
                # else:
                #     # Otherwise, check whether to freeze goal cost
                #     if min_val == 0 and min_index < size_val[1]:
                #         obj_val_np[i, min_index:] = 0
                #     elif max_val >= 100 and max_index < size_val[1]:
                #         obj_val_np[i, max_index:] = 100

                # if min_val == 0 and min_index < size_val[1]:
                #     obj_val_np[i, min_index:] = 0
        except ValueError:
            print("No objective values!\n\n\n")
            pass

        return tf.constant(obj_val_np, dtype=tf.float32)

    # def _freeze_cost(self, objective_values):
    #
    #     obj_val_np = objective_values.numpy()
    #
    #     size_val = obj_val_np.shape
    #
    #     try:
    #         for i in range(size_val[0]):
    #             min_val = np.amin(obj_val_np[i, :])
    #             min_index = np.argmin(obj_val_np[i, :])
    #             max_val = np.amax(obj_val_np[i, :])
    #             max_index = np.argmax(obj_val_np[i, :])
    #
    #             # Freeze both obstacle enter and goal reaching cost
    #             # If max pos first
    #             if max_index < min_index:
    #                 # First check whether to freeze obstacle cost
    #                 if max_val >= 100 and max_index < size_val[1]:
    #                     obj_val_np[i, max_index:] = 100
    #                 elif min_val == 0 and min_index < size_val[1]:
    #                     obj_val_np[i, min_index:] = 0
    #             else:
    #                 # Otherwise, check whether to freeze goal cost
    #                 if min_val == 0 and min_index < size_val[1]:
    #                     obj_val_np[i, min_index:] = 0
    #                 elif max_val >= 100 and max_index < size_val[1]:
    #                     obj_val_np[i, max_index:] = 100
    #
    #             # if min_val == 0 and min_index < size_val[1]:
    #             #     obj_val_np[i, min_index:] = 0
    #     except ValueError:
    #         print("No objective values!\n\n\n")
    #         pass
    #
    #     return tf.constant(obj_val_np, dtype=tf.float32)