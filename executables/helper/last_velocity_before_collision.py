import pickle
import numpy as np
import os

"""
Compute velocity at the last time step before collision
"""

# trajectory folder
traj_folder = '/media/anjianl/anjian/cs-mars-04/projects/WayPtNav/useful_result/session_2019-10-01_20-21-37_pretrained_ctrlhorizon_0.25_500maps/rgb_resnet50_nn_waypoint_simulator/trajectories'

num_files = 500
collision_num = 0
speed = 0

for j in range(num_files):

    if j % 50 == 1:
        print('processing', j, 'files')

    # Find the filename
    filename = os.path.join(traj_folder, 'traj_%i.pkl' % (j))

    if not os.path.exists(filename):
        continue

    # Load the file
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)

    if data['episode_type_string'] == 'Collision':
        collision_num += 1
        size = data['vehicle_trajectory']['speed_nk1'].shape[1]
        speed += data['vehicle_trajectory']['speed_nk1'][0, size-1, 0]

print('collision num is', collision_num)
print('average speed before collision is', speed / collision_num)
