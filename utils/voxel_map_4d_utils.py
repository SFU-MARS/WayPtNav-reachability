import tensorflow as tf
import numpy as np


class VoxelMap4d(object):
    """
    A 4d voxel map, storing reachability information.
    Used for interpolation in cost function.

    """

    def __init__(self, dx, dy, dtheta, dv, origin_4, map_size_4, function_array_4d=None):
        self.dx = tf.constant(dx, dtype=tf.float32)
        self.dy = tf.constant(dy, dtype=tf.float32)
        self.dtheta = tf.constant(dtheta, dtype=tf.float32)
        self.dv = tf.constant(dv, dtype=tf.float32)
        self.map_origin_4 = tf.constant(origin_4, dtype=tf.float32)
        self.map_size_int32_4 = tf.constant(map_size_4, dtype=tf.int32)
        self.map_size_float32_4 = tf.constant(map_size_4, dtype=tf.float32)
        self.voxel_function_4d = function_array_4d

    def compute_voxel_function(self, position_nk2, heading_nk1, speed_nk1, invalid_value=100.):
        voxel_space_position_nk1_x, voxel_space_position_nk1_y, voxel_space_heading_nk1, voxel_space_speed_nk1 \
            = self.grid_world_to_voxel_world(position_nk2, heading_nk1, speed_nk1)

        # Define the lower and upper voxels. Mod is done to make sure that the invalid voxels have been
        # assigned a valid voxel. However, these invalid voxels will be discarded later.
        lower_voxel_indices_nk1_x = tf.mod(tf.cast(tf.floor(voxel_space_position_nk1_x), tf.int32),
                                           self.map_size_int32_4[0])
        upper_voxel_indices_nk1_x = tf.mod(lower_voxel_indices_nk1_x + 1, self.map_size_int32_4[0])

        lower_voxel_indices_nk1_y = tf.mod(tf.cast(tf.floor(voxel_space_position_nk1_y), tf.int32),
                                           self.map_size_int32_4[1])
        upper_voxel_indices_nk1_y = tf.mod(lower_voxel_indices_nk1_y + 1, self.map_size_int32_4[1])

        lower_voxel_indices_nk1_theta = tf.mod(tf.cast(tf.floor(voxel_space_heading_nk1), tf.int32),
                                               self.map_size_int32_4[2])
        upper_voxel_indices_nk1_theta = tf.mod(lower_voxel_indices_nk1_theta + 1, self.map_size_int32_4[2])

        lower_voxel_indices_nk1_v = tf.mod(tf.cast(tf.floor(voxel_space_speed_nk1), tf.int32),
                                           self.map_size_int32_4[3])
        upper_voxel_indices_nk1_v = tf.mod(lower_voxel_indices_nk1_v + 1, self.map_size_int32_4[3])

        # to float
        lower_voxel_float_x = tf.cast(lower_voxel_indices_nk1_x, dtype=tf.float32)
        lower_voxel_float_y = tf.cast(lower_voxel_indices_nk1_y, dtype=tf.float32)
        lower_voxel_float_theta = tf.cast(lower_voxel_indices_nk1_theta, dtype=tf.float32)
        lower_voxel_float_v = tf.cast(lower_voxel_indices_nk1_v, dtype=tf.float32)
        upper_voxel_float_x = tf.cast(upper_voxel_indices_nk1_x, dtype=tf.float32)
        upper_voxel_float_y = tf.cast(upper_voxel_indices_nk1_y, dtype=tf.float32)
        upper_voxel_float_theta = tf.cast(upper_voxel_indices_nk1_theta, dtype=tf.float32)
        upper_voxel_float_v = tf.cast(upper_voxel_indices_nk1_v, dtype=tf.float32)

        # Voxel indices for 8 corner voxels. All in x, y
        voxel_index_8 = tf.concat([lower_voxel_indices_nk1_x, upper_voxel_indices_nk1_x,
                                   lower_voxel_indices_nk1_y, upper_voxel_indices_nk1_y,
                                   lower_voxel_indices_nk1_theta, upper_voxel_indices_nk1_theta,
                                   lower_voxel_indices_nk1_v, upper_voxel_indices_nk1_v], axis=2)

        voxel_index_int64_8 = tf.cast(voxel_index_8, dtype=tf.int64)

        # Voxel function values at corner points
        # 4d linear interpolation (Based on Trilinear interpolation, add one more dimension for v )
        # Thus, there are 2 parts. First part is for lower v, second part is for upper v.
        data0000 = tf.gather_nd(self.voxel_function_4d, tf.gather(voxel_index_int64_8, [0, 2, 4, 6], axis=2))
        data1000 = tf.gather_nd(self.voxel_function_4d, tf.gather(voxel_index_int64_8, [1, 2, 4, 6], axis=2))
        data0010 = tf.gather_nd(self.voxel_function_4d, tf.gather(voxel_index_int64_8, [0, 2, 5, 6], axis=2))
        data1010 = tf.gather_nd(self.voxel_function_4d, tf.gather(voxel_index_int64_8, [1, 2, 5, 6], axis=2))
        data0100 = tf.gather_nd(self.voxel_function_4d, tf.gather(voxel_index_int64_8, [0, 3, 4, 6], axis=2))
        data1100 = tf.gather_nd(self.voxel_function_4d, tf.gather(voxel_index_int64_8, [1, 3, 4, 6], axis=2))
        data0110 = tf.gather_nd(self.voxel_function_4d, tf.gather(voxel_index_int64_8, [0, 3, 5, 6], axis=2))
        data1110 = tf.gather_nd(self.voxel_function_4d, tf.gather(voxel_index_int64_8, [1, 3, 5, 6], axis=2))

        data0001 = tf.gather_nd(self.voxel_function_4d, tf.gather(voxel_index_int64_8, [0, 2, 4, 7], axis=2))
        data1001 = tf.gather_nd(self.voxel_function_4d, tf.gather(voxel_index_int64_8, [1, 2, 4, 7], axis=2))
        data0011 = tf.gather_nd(self.voxel_function_4d, tf.gather(voxel_index_int64_8, [0, 2, 5, 7], axis=2))
        data1011 = tf.gather_nd(self.voxel_function_4d, tf.gather(voxel_index_int64_8, [1, 2, 5, 7], axis=2))
        data0101 = tf.gather_nd(self.voxel_function_4d, tf.gather(voxel_index_int64_8, [0, 3, 4, 7], axis=2))
        data1101 = tf.gather_nd(self.voxel_function_4d, tf.gather(voxel_index_int64_8, [1, 3, 4, 7], axis=2))
        data0111 = tf.gather_nd(self.voxel_function_4d, tf.gather(voxel_index_int64_8, [0, 3, 5, 7], axis=2))
        data1111 = tf.gather_nd(self.voxel_function_4d, tf.gather(voxel_index_int64_8, [1, 3, 5, 7], axis=2))

        # Define alphas for x interpolation
        alpha1 = upper_voxel_float_x[:, :, 0] - voxel_space_position_nk1_x[:, :, 0]
        alpha2 = voxel_space_position_nk1_x[:, :, 0] - lower_voxel_float_x[:, :, 0]

        # Define betas for y interpolation
        betas1 = upper_voxel_float_y[:, :, 0] - voxel_space_position_nk1_y[:, :, 0]
        betas2 = voxel_space_position_nk1_y[:, :, 0] - lower_voxel_float_y[:, :, 0]

        # Define gammas for theta interpolation
        gammas1 = upper_voxel_float_theta[:, :, 0] - voxel_space_heading_nk1[:, :, 0]
        gammas2 = voxel_space_heading_nk1[:, :, 0] - lower_voxel_float_theta[:, :, 0]

        # Define delta for v interpolation
        delta1 = upper_voxel_float_v[:, :, 0] - voxel_space_speed_nk1[:, :, 0]
        delta2 = voxel_space_speed_nk1[:, :, 0] - lower_voxel_float_v[:, :, 0]

        # Interpolation in the x-direction
        data000 = data0000 * alpha1 + data1000 * alpha2
        data010 = data0010 * alpha1 + data1010 * alpha2
        data100 = data0100 * alpha1 + data1100 * alpha2
        data110 = data0110 * alpha1 + data1110 * alpha2

        data001 = data0001 * alpha1 + data1001 * alpha2
        data011 = data0011 * alpha1 + data1011 * alpha2
        data101 = data0101 * alpha1 + data1101 * alpha2
        data111 = data0111 * alpha1 + data1111 * alpha2

        # Interpolation in the y-direction
        data00 = data000 * betas1 + data100 * betas2
        data10 = data010 * betas1 + data110 * betas2

        data01 = data001 * betas1 + data101 * betas2
        data11 = data011 * betas1 + data111 * betas2

        # Interpolation in the theta-direction
        data0 = data00 * gammas1 + data10 * gammas2

        data1 = data01 * gammas1 + data11 * gammas2

        # Interpolation in the v-direction
        data = data0 * delta1 + data1 * delta2

        valid_voxels_4d = self.is_valid_voxel(position_nk2, heading_nk1, speed_nk1)

        return tf.where(valid_voxels_4d, data, tf.ones_like(data) * invalid_value)

    def grid_world_to_voxel_world(self, position_nk2, heading_nk1, speed_nk1):
        """
        Convert the positions in the global world to the voxel world coordinates.
        """

        index_nk1_x = (position_nk2[:, :, 0] - self.map_origin_4[0]) / self.dx
        index_nk1_y = (position_nk2[:, :, 1] - self.map_origin_4[1]) / self.dy
        index_nk1_x = tf.expand_dims(index_nk1_x, 2)  # Maintain a 3d shape
        index_nk1_y = tf.expand_dims(index_nk1_y, 2)  # Maintain a 3d shape

        index_nk1_theta = (heading_nk1 - self.map_origin_4[2]) / self.dtheta

        index_nk1_v = (speed_nk1 - self.map_origin_4[3]) / self.dv

        return index_nk1_x, index_nk1_y, index_nk1_theta, index_nk1_v

    def is_valid_voxel(self, position_nk2, heading_nk1, speed_nk1):
        """
        Check if a given set of positions and headings are within the voxel map or not.
        """
        voxel_space_position_nk1_x, voxel_space_position_nk1_y, voxel_space_heading_nk1, voxel_space_speed_nk1 \
            = self.grid_world_to_voxel_world(position_nk2, heading_nk1, speed_nk1)

        valid_x = tf.logical_and(tf.keras.backend.all(voxel_space_position_nk1_x >= 0., axis=2),
                                 tf.keras.backend.all(voxel_space_position_nk1_x < (self.map_size_float32_4[0] - 1.),
                                                      axis=2))

        valid_y = tf.logical_and(tf.keras.backend.all(voxel_space_position_nk1_y >= 0., axis=2),
                                 tf.keras.backend.all(voxel_space_position_nk1_y < (self.map_size_float32_4[1] - 1.),
                                                      axis=2))

        valid_theta = tf.logical_and(tf.keras.backend.all(voxel_space_heading_nk1 >= 0., axis=2),
                                     tf.keras.backend.all(voxel_space_heading_nk1 < (self.map_size_float32_4[2] - 1.),
                                                          axis=2))

        valid_v = tf.logical_and(tf.keras.backend.all(voxel_space_speed_nk1 >= 0., axis=2),
                                 tf.keras.backend.all(voxel_space_speed_nk1 < (self.map_size_float32_4[3] - 1.),
                                                      axis=2))

        valid_1 = tf.logical_and(valid_x, valid_y)
        valid_2 = tf.logical_and(valid_1, valid_theta)
        valid_3 = tf.logical_and(valid_2, valid_v)

        return valid_3
