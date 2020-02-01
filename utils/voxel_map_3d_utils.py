import tensorflow as tf
import numpy as np


class VoxelMap3d(object):
    """
    A 3d voxel map, storing reachability information.
    Used for interpolation in cost function.

    """

    def __init__(self, dx, dy, dtheta, origin_3, map_size_3, function_array_3d=None):
        """
        Property of 3d voxel map

        :param dx:
        :param dy:
        :param dtheta:
        :param origin_3:
        :param map_size_3:
        :param function_array_3d:
        """
        self.dx = tf.constant(dx, dtype=tf.float32)
        self.dy = tf.constant(dy, dtype=tf.float32)
        self.dtheta = tf.constant(dtheta, dtype=tf.float32)
        self.map_origin_3 = tf.constant(origin_3, dtype=tf.float32)
        self.map_size_int32_3 = tf.constant(map_size_3, dtype=tf.int32)
        self.map_size_float32_3 = tf.constant(map_size_3, dtype=tf.float32)
        self.voxel_function_3d = function_array_3d

    def compute_voxel_function(self, position_nk2, heading_nk1, invalid_value=100.):
        """
        Interpolate trajectory on 3d voxel map

        :param position_nk2:
        :param heading_nk1:
        :param invalid_value:
        :return:
        """
        voxel_space_position_nk1_x, voxel_space_position_nk1_y, voxel_space_heading_nk1 \
            = self.grid_world_to_voxel_world(position_nk2, heading_nk1)

        # Define the lower and upper voxels. Mod is done to make sure that the invalid voxels have been
        # assigned a valid voxel. However, these invalid voxels will be discarded later.
        lower_voxel_indices_nk1_x = tf.mod(tf.cast(tf.floor(voxel_space_position_nk1_x), tf.int32),
                                           self.map_size_int32_3[0])
        upper_voxel_indices_nk1_x = tf.mod(lower_voxel_indices_nk1_x + 1, self.map_size_int32_3[0])

        lower_voxel_indices_nk1_y = tf.mod(tf.cast(tf.floor(voxel_space_position_nk1_y), tf.int32),
                                           self.map_size_int32_3[1])
        upper_voxel_indices_nk1_y = tf.mod(lower_voxel_indices_nk1_y + 1, self.map_size_int32_3[1])

        lower_voxel_indices_nk1_theta = tf.mod(tf.cast(tf.floor(voxel_space_heading_nk1), tf.int32),
                                               self.map_size_int32_3[2])
        upper_voxel_indices_nk1_theta = tf.mod(lower_voxel_indices_nk1_theta + 1, self.map_size_int32_3[2])

        # to float
        lower_voxel_float_x = tf.cast(lower_voxel_indices_nk1_x, dtype=tf.float32)
        lower_voxel_float_y = tf.cast(lower_voxel_indices_nk1_y, dtype=tf.float32)
        lower_voxel_float_theta = tf.cast(lower_voxel_indices_nk1_theta, dtype=tf.float32)
        upper_voxel_float_x = tf.cast(upper_voxel_indices_nk1_x, dtype=tf.float32)
        upper_voxel_float_y = tf.cast(upper_voxel_indices_nk1_y, dtype=tf.float32)
        upper_voxel_float_theta = tf.cast(upper_voxel_indices_nk1_theta, dtype=tf.float32)

        # Voxel indices for 6 corner voxels. All in x, y
        voxel_index_6 = tf.concat([lower_voxel_indices_nk1_x, upper_voxel_indices_nk1_x,
                                   lower_voxel_indices_nk1_y, upper_voxel_indices_nk1_y,
                                   lower_voxel_indices_nk1_theta, upper_voxel_indices_nk1_theta], axis=2)

        voxel_index_int64_6 = tf.cast(voxel_index_6, dtype=tf.int64)

        # Voxel function values at corner points
        # notation refer to wikipedia, trilinear interpolation
        # https://en.wikipedia.org/wiki/Trilinear_interpolation
        data000 = tf.gather_nd(self.voxel_function_3d, tf.gather(voxel_index_int64_6, [0, 2, 4], axis=2))
        data100 = tf.gather_nd(self.voxel_function_3d, tf.gather(voxel_index_int64_6, [1, 2, 4], axis=2))
        data001 = tf.gather_nd(self.voxel_function_3d, tf.gather(voxel_index_int64_6, [0, 2, 5], axis=2))
        data101 = tf.gather_nd(self.voxel_function_3d, tf.gather(voxel_index_int64_6, [1, 2, 5], axis=2))
        data010 = tf.gather_nd(self.voxel_function_3d, tf.gather(voxel_index_int64_6, [0, 3, 4], axis=2))
        data110 = tf.gather_nd(self.voxel_function_3d, tf.gather(voxel_index_int64_6, [1, 3, 4], axis=2))
        data011 = tf.gather_nd(self.voxel_function_3d, tf.gather(voxel_index_int64_6, [0, 3, 5], axis=2))
        data111 = tf.gather_nd(self.voxel_function_3d, tf.gather(voxel_index_int64_6, [1, 3, 5], axis=2))

        # Define alphas for x interpolation
        alpha1 = upper_voxel_float_x[:, :, 0] - voxel_space_position_nk1_x[:, :, 0]
        alpha2 = voxel_space_position_nk1_x[:, :, 0] - lower_voxel_float_x[:, :, 0]

        # Define betas for y interpolation
        betas1 = upper_voxel_float_y[:, :, 0] - voxel_space_position_nk1_y[:, :, 0]
        betas2 = voxel_space_position_nk1_y[:, :, 0] - lower_voxel_float_y[:, :, 0]

        # Define gammas for theta interpolation
        gammas1 = upper_voxel_float_theta[:, :, 0] - voxel_space_heading_nk1[:, :, 0]
        gammas2 = voxel_space_heading_nk1[:, :, 0] - lower_voxel_float_theta[:, :, 0]

        # Interpolation in the x-direction
        data00 = data000 * alpha1 + data100 * alpha2
        data01 = data001 * alpha1 + data101 * alpha2
        data10 = data010 * alpha1 + data110 * alpha2
        data11 = data011 * alpha1 + data111 * alpha2

        # Interpolation in the y-direction
        data0 = data00 * betas1 + data10 * betas2
        data1 = data01 * betas1 + data11 * betas2

        # Interpolation in the theta-direction
        data = data0 * gammas1 + data1 * gammas2

        valid_voxels_3d = self.is_valid_voxel(position_nk2, heading_nk1)

        return tf.where(valid_voxels_3d, data, tf.ones_like(data) * invalid_value)

    def grid_world_to_voxel_world(self, position_nk2, heading_nk1):
        """
        Convert the positions in the global world to the voxel world coordinates.
        """

        position_nk1_x = (position_nk2[:, :, 0] - self.map_origin_3[0]) / self.dx
        position_nk1_y = (position_nk2[:, :, 1] - self.map_origin_3[1]) / self.dy
        position_nk1_x = tf.expand_dims(position_nk1_x, 2) # Maintain a 3d shape
        position_nk1_y = tf.expand_dims(position_nk1_y, 2) # Maintain a 3d shape
        heading_nk1 = (heading_nk1 + np.pi) / self.dtheta

        return position_nk1_x, position_nk1_y, heading_nk1

    def is_valid_voxel(self, position_nk2, heading_nk1):
        """
        Check if a given set of positions and headings are within the voxel map or not.

        """
        voxel_space_position_nk1_x, voxel_space_position_nk1_y, voxel_space_heading_nk1 \
            = self.grid_world_to_voxel_world(position_nk2, heading_nk1)

        valid_x = tf.logical_and(tf.keras.backend.all(voxel_space_position_nk1_x >= 0., axis=2),
                                 tf.keras.backend.all(voxel_space_position_nk1_x < (self.map_size_float32_3[0] - 1.),
                                                      axis=2))

        valid_y = tf.logical_and(tf.keras.backend.all(voxel_space_position_nk1_y >= 0., axis=2),
                                 tf.keras.backend.all(voxel_space_position_nk1_y < (self.map_size_float32_3[1] - 1.),
                                                      axis=2))

        valid_theta = tf.logical_and(tf.keras.backend.all(voxel_space_heading_nk1 >= 0., axis=2),
                                     tf.keras.backend.all(voxel_space_heading_nk1 < (self.map_size_float32_3[2] - 1.),
                                                          axis=2))

        valid_1 = tf.logical_and(valid_x, valid_y)
        valid_2 = tf.logical_and(valid_1, valid_theta)

        return valid_2
