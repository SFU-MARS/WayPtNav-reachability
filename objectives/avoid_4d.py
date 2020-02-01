import tensorflow as tf
import numpy as np
from objectives.objective_function import Objective


class Avoid4d(Objective):
    """
    Avoid cost measured on 4D avoid TTC value map
    """

    def __init__(self, params, reachability_map):
        self.p = params
        self.reachability_map = reachability_map
        self.tag = 'avoid_4d'
        self.avoid_4d_ttr_scale = self.p.avoid_4d_ttr_scale  # The whole cost is reach_avoid_TTR + scale * avoid_TTR

    def compute_avoid_4d(self, trajectory):

        return self.reachability_map.avoid_4d_map.compute_voxel_function(trajectory.position_nk2(),
                                                                         trajectory.heading_nk1(),
                                                                         trajectory.speed_nk1())

    def evaluate_objective(self, trajectory):

        # Value indicates the time to enter the obstacles. 100 means never enter the obstacles
        avoid_4d = self.compute_avoid_4d(trajectory)

        largest_ttr = tf.constant(100, dtype=tf.float32)

        # 100 - avoid value:, 0 means never enter the obstacles. 100 means in the obstacles
        avoid_4d_negative = tf.subtract(largest_ttr, avoid_4d)

        # If freezing the cost:
        # try:
        #     avoid_4d_negative = self._freeze_cost_obstacle_enter(avoid_4d_negative)
        # except ValueError:
        #     print("cannot freeze_cost_obstacle_enter in avoid 4d")

        return self.avoid_4d_ttr_scale * avoid_4d_negative

    def _freeze_cost_obstacle_enter(self, objective_values):

        obj_val_np = objective_values.numpy()

        size_val = obj_val_np.shape

        for i in range(size_val[0]):
            max_val = np.amax(obj_val_np[i, :])
            max_index = np.argmax(obj_val_np[i, :])

            if max_val >= 100 and max_index < size_val[1]:
                obj_val_np[i, max_index:] = 100

        return tf.constant(obj_val_np, dtype=tf.float32)