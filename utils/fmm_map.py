import tensorflow as tf
import numpy as np
import skfmm
import sys
if sys.version[0] == '2':
    from voxel_map_utils import VoxelMap
else:  # python3
    from utils.voxel_map_utils import VoxelMap


class FmmMap(object):
    """
    Maintain a FMM distance and angle map corresponding to a given goal and occupancy grid.
    """

    def __init__(self, goal_grid_mn, dx=1, map_origin_2=tf.zeros([2], dtype=tf.float32), mask_grid_mn=None):
        """
        Args:
            goal_grid_mn: A mxn grid containing the goal positions. Typically, it should have 0s at the goal positions
                          and 1 everywhere else. In general, the goal should be defined as the subzero level set.
            dx: The step size in the goal grid.
            map_origin_2: The origin of the goal grid.
            mask_grid_mn: The part of the goal array to be masked before computing the fmm distance. Typically, the
                          array should have 1 at the grid points to be masked and 0 everywhere else.
        """
        m, n = goal_grid_mn.shape[0], goal_grid_mn.shape[1]
        self.mask_grid_mn = mask_grid_mn
        self.goal_grid_mn = goal_grid_mn
        self.map_origin_2 = map_origin_2
        self.dx = dx

        self.fmm_distance_map = VoxelMap(scale=dx,
                                         origin_2=map_origin_2,
                                         map_size_2=tf.constant([n, m], dtype=tf.float32),
                                         function_array_mn=None)
        self.fmm_angle_map = VoxelMap(scale=dx,
                                      origin_2=map_origin_2,
                                      map_size_2=tf.constant([n, m], dtype=tf.float32),
                                      function_array_mn=None)
        self.compute_fmm_distance_and_angle()

    def compute_fmm_distance_and_angle(self, mask_value=1000):
        """
        Compute the fmm distance based on the goal array and mask array.

        """
        # Mask the goal array
        if self.mask_grid_mn is not None:
            phi = np.ma.MaskedArray(self.goal_grid_mn, self.mask_grid_mn) # mask the obstacle_occupancy_grid invalid
        else:
            phi = self.goal_grid_mn

        # Compute the fmm distance
        fmm_distance = skfmm.distance(phi, dx=self.fmm_distance_map.map_scale*np.ones(2))
        # Return the signed distance from the *zero contour* of the array phi.
        # in phi, goal: -1, else 1, obstacle being masked

        # Assign some distance at the mask
        if self.mask_grid_mn is not None:
            fmm_distance = fmm_distance.filled(mask_value)
            # Return a copy of self, with masked values filled with a given value.
            #         **However**, if there are no masked values to fill, self will be
            #         returned instead as an ndarray.
            # Here, Mask obstacles to be 1000, and goals to be 1, and elsewhere, sign-distance to goals

        # Compute the fmm angle
        gradient_y, gradient_x = np.gradient(fmm_distance, self.fmm_distance_map.map_scale) # Return the gradient of an N-dimensional array
        fmm_angle = np.arctan2(-gradient_y, -gradient_x) # use arctan(y/x) to compute angle

        # Assign fmm distance map and angle
        self.fmm_distance_map.voxel_function_mn = tf.constant(fmm_distance, dtype=tf.float32) # Assign the distance map to fmm_distance_map object, used in compute_voxel_function
        self.fmm_angle_map.voxel_function_mn = tf.constant(fmm_angle, dtype=tf.float32) # Assign the angle to fmm_angle_map object, used in compute_voxel_function

    def change_goal(self, goal_positions_n2, mask_value=1000):
        """
        Recompute the fmm maps based on the new goal position.
        """
        # print("change fmm map!")
        map_size_2 = self.goal_grid_mn.shape[::-1]
        goal_array_mn = self._create_fmm_map_goal_array_mn(goal_positions_n2,
                                                           map_size_2, self.dx,
                                                           self.map_origin_2,
                                                           self.mask_grid_mn)
        self.goal_grid_mn = goal_array_mn
        self.compute_fmm_distance_and_angle(mask_value)

    def render_distance_map(self, ax):
        ax.contour(self.fmm_distance_map.voxel_function_mn)
        ax.set_title('Fmm Dist')

    def render_angle_map(self, ax):
        ax.contour(self.fmm_angle_map.voxel_function_mn)
        ax.set_title('Fmm Angle')

    @staticmethod
    def _create_fmm_map_goal_array_mn(goal_positions_n2, map_size_2, dx=1,
                                      map_origin_2=tf.zeros([2], dtype=tf.float32),
                                      mask_grid_mn=None):
        goal_array_mn = np.ones((map_size_2[1], map_size_2[0]))
        goal_index_x = np.floor((goal_positions_n2[:, 0] / dx) - map_origin_2[0]).astype(np.int32)
        goal_index_y = np.floor((goal_positions_n2[:, 1] / dx) - map_origin_2[1]).astype(np.int32)
        goal_array_mn[goal_index_y, goal_index_x] = -1.
        return goal_array_mn

    @classmethod
    def create_fmm_map_based_on_goal_position(cls, goal_positions_n2, map_size_2, dx=1,
                                              map_origin_2=tf.zeros([2], dtype=tf.float32),
                                              mask_grid_mn=None):
        """
        Create a fmm map based on a given goal position.
        """
        goal_array_mn = FmmMap._create_fmm_map_goal_array_mn(goal_positions_n2,
                                                             map_size_2, dx,
                                                             map_origin_2,
                                                             mask_grid_mn)
        return cls(goal_grid_mn=goal_array_mn,
                   dx=dx,
                   map_origin_2=map_origin_2,
                   mask_grid_mn=mask_grid_mn)
