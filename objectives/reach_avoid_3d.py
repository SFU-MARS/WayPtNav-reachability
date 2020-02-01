import tensorflow as tf
import numpy as np
from objectives.objective_function import Objective


class ReachAvoid3d(Objective):
    """
    Reach avoid cost measured on 3D reach-avoid TTC value map
    """

    def __init__(self, params, reachability_map):
        self.p = params
        self.reachability_map = reachability_map
        self.tag = 'reach_avoid_3d'

    def compute_reach_avoid_3d(self, trajectory):

        return self.reachability_map.reach_avoid_3d_map.compute_voxel_function(trajectory.position_nk2(), trajectory.heading_nk1())

    def evaluate_objective(self, trajectory):
        reach_avoid_3d = self.compute_reach_avoid_3d(trajectory)
        # reach_avoid_3d = self._freeze_cost(reach_avoid_3d)
        return reach_avoid_3d

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
    #             if min_val == 0 and min_index < size_val[1]:
    #                 # tf.assign(a[i, min_index:], 0)
    #                 obj_val_np[i, min_index:] = 0
    #     except ValueError:
    #         print("No objective values!\n\n\n")
    #         pass
    #
    #     return tf.constant(obj_val_np, dtype=tf.float32)

        # It takes much more time to execute in tf
        #
        # size_obj_val = np.asarray(tf.shape(objective_values))
        # objective_values_freeze = tf.constant(tf.zeros(size_obj_val[1]), dtype=tf.float32)
        #
        # for i in range(size_obj_val[0]):
        #     min_val = tf.reduce_min(objective_values[i, :])
        #     min_index = tf.argmin(objective_values[i, :])
        #     if tf.equal(min_val, 0) and tf.less(min_index, size_obj_val[1]):
        #         freeze_cost = tf.constant(tf.zeros(size_obj_val[1] - min_index), dtype=tf.float32)
        #         a_curr = tf.concat([objective_values[i, :min_index], freeze_cost], axis=0)
        #     else:
        #         a_curr = tf.constant(objective_values[i, :], dtype=tf.float32)
        #     # a_new.append(a_curr)
        #     objective_values_freeze = tf.concat([objective_values_freeze, a_curr], 0)
        #
        # # a_final = tf.stack(a_new)
        # objective_values_freeze = tf.reshape(objective_values_freeze[size_obj_val[1]:], size_obj_val)
        #
        # return objective_values_freeze

