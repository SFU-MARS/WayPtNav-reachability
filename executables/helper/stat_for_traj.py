import pickle
import numpy as np
import os

"""
Compute the statistics. Among results for 4 cases: reachability vs oldcost, 1.5 vs 0.25 control horizon, we compute the 
amount of episodes of 16 different combinations.
"""


# reachability 1.5
traj_1_folder = '/media/anjianl/anjian/cs-mars-04/projects/WayPtNav/useful_result/session_2019-11-22_15-21-08_lr-4_reg-6_ckpt19_v2_ctrlhorizon_1.5_500maps/rgb_resnet50_nn_waypoint_simulator/trajectories'
# reachability 0.25
traj_2_folder = '/media/anjianl/anjian/cs-mars-04/projects/WayPtNav/useful_result/session_2019-10-02_19-58-42_lr-4_reg-6_ckpt19_v2_ctrlhorizon_0.25_500maps/rgb_resnet50_nn_waypoint_simulator/trajectories'
# old cost 1.5
traj_3_folder = '/media/anjianl/anjian/cs-mars-04/projects/WayPtNav/useful_result/session_2019-11-22_16-18-00_pretrained_ctrlhorizon_1.5_500maps/rgb_resnet50_nn_waypoint_simulator/trajectories'
# old cost 0.25
traj_4_folder = '/media/anjianl/anjian/cs-mars-04/projects/WayPtNav/useful_result/session_2019-10-01_20-21-37_pretrained_ctrlhorizon_0.25_500maps/rgb_resnet50_nn_waypoint_simulator/trajectories'

num_files = 500
real_num_files = 0

gggg_num = 0
gggr_num = 0
ggrg_num = 0
ggrr_num = 0

grgg_num = 0
grgr_num = 0
grrg_num = 0
grrr_num = 0

rggg_num = 0
rggr_num = 0
rgrg_num = 0
rgrr_num = 0

rrgg_num = 0
rrgr_num = 0
rrrg_num = 0
rrrr_num = 0

for j in range(num_files):

    if j % 20 == 1:
        print('processing', j, 'files')

    # Find the filename
    filename_1 = os.path.join(traj_1_folder, 'traj_%i.pkl' % (j))
    filename_2 = os.path.join(traj_2_folder, 'traj_%i.pkl' % (j))
    filename_3 = os.path.join(traj_3_folder, 'traj_%i.pkl' % (j))
    filename_4 = os.path.join(traj_4_folder, 'traj_%i.pkl' % (j))

    if (not os.path.exists(filename_1)) or (not os.path.exists(filename_2)) or (not os.path.exists(filename_3)) or (not os.path.exists(filename_4)):
        continue

    real_num_files += 1
    
    # Load the file
    with open(filename_1, 'rb') as handle:
        data_1 = pickle.load(handle)
    with open(filename_2, 'rb') as handle:
        data_2 = pickle.load(handle)
    with open(filename_3, 'rb') as handle:
        data_3 = pickle.load(handle)
    with open(filename_4, 'rb') as handle:
        data_4 = pickle.load(handle)

    if data_1['episode_type_string'] == 'Success' and data_2['episode_type_string'] == 'Success' and \
            data_3['episode_type_string'] == 'Success' and data_4['episode_type_string'] == 'Success':
        gggg_num = gggg_num + 1
    if data_1['episode_type_string'] == 'Success' and data_2['episode_type_string'] == 'Success' and \
            data_3['episode_type_string'] == 'Success' and data_4['episode_type_string'] != 'Success':
        gggr_num = gggr_num + 1
    if data_1['episode_type_string'] == 'Success' and data_2['episode_type_string'] == 'Success' and \
            data_3['episode_type_string'] != 'Success' and data_4['episode_type_string'] == 'Success':
        ggrg_num = ggrg_num + 1
    if data_1['episode_type_string'] == 'Success' and data_2['episode_type_string'] == 'Success' and \
            data_3['episode_type_string'] != 'Success' and data_4['episode_type_string'] != 'Success':
        ggrr_num = ggrr_num + 1

    if data_1['episode_type_string'] == 'Success' and data_2['episode_type_string'] != 'Success' and \
            data_3['episode_type_string'] == 'Success' and data_4['episode_type_string'] == 'Success':
        grgg_num = grgg_num + 1
    if data_1['episode_type_string'] == 'Success' and data_2['episode_type_string'] != 'Success' and \
            data_3['episode_type_string'] == 'Success' and data_4['episode_type_string'] != 'Success':
        grgr_num = grgr_num + 1
    if data_1['episode_type_string'] == 'Success' and data_2['episode_type_string'] != 'Success' and \
            data_3['episode_type_string'] != 'Success' and data_4['episode_type_string'] == 'Success':
        grrg_num = grrg_num + 1
    if data_1['episode_type_string'] == 'Success' and data_2['episode_type_string'] != 'Success' and \
            data_3['episode_type_string'] != 'Success' and data_4['episode_type_string'] != 'Success':
        grrr_num = grrr_num + 1

    if data_1['episode_type_string'] != 'Success' and data_2['episode_type_string'] == 'Success' and \
            data_3['episode_type_string'] == 'Success' and data_4['episode_type_string'] == 'Success':
        rggg_num = rggg_num + 1
    if data_1['episode_type_string'] != 'Success' and data_2['episode_type_string'] == 'Success' and \
            data_3['episode_type_string'] == 'Success' and data_4['episode_type_string'] != 'Success':
        rggr_num = rggr_num + 1
    if data_1['episode_type_string'] != 'Success' and data_2['episode_type_string'] == 'Success' and \
            data_3['episode_type_string'] != 'Success' and data_4['episode_type_string'] == 'Success':
        rgrg_num = rgrg_num + 1
    if data_1['episode_type_string'] != 'Success' and data_2['episode_type_string'] == 'Success' and \
            data_3['episode_type_string'] != 'Success' and data_4['episode_type_string'] != 'Success':
        rgrr_num = rgrr_num + 1

    if data_1['episode_type_string'] != 'Success' and data_2['episode_type_string'] != 'Success' and \
            data_3['episode_type_string'] == 'Success' and data_4['episode_type_string'] == 'Success':
        rrgg_num = rrgg_num + 1
    if data_1['episode_type_string'] != 'Success' and data_2['episode_type_string'] != 'Success' and \
            data_3['episode_type_string'] == 'Success' and data_4['episode_type_string'] != 'Success':
        rrgr_num = rrgr_num + 1
    if data_1['episode_type_string'] != 'Success' and data_2['episode_type_string'] != 'Success' and \
            data_3['episode_type_string'] != 'Success' and data_4['episode_type_string'] == 'Success':
        rrrg_num = rrrg_num + 1
    if data_1['episode_type_string'] != 'Success' and data_2['episode_type_string'] != 'Success' and \
            data_3['episode_type_string'] != 'Success' and data_4['episode_type_string'] != 'Success':
        rrrr_num = rrrr_num + 1

print("real num file is", real_num_files)
print("gggg_num has ", gggg_num, "the rate is", gggg_num / real_num_files)
print("gggr_num has ", gggr_num, "the rate is", gggr_num / real_num_files)
print("ggrg_num has ", ggrg_num, "the rate is", ggrg_num / real_num_files)
print("ggrr_num has ", ggrr_num, "the rate is", ggrr_num / real_num_files)

print("grgg_num has ", grgg_num, "the rate is", grgg_num / real_num_files)
print("grgr_num has ", grgr_num, "the rate is", grgr_num / real_num_files)
print("grrg_num has ", grrg_num, "the rate is", grrg_num / real_num_files)
print("grrr_num has ", grrr_num, "the rate is", grrr_num / real_num_files)

print("rggg_num has ", rggg_num, "the rate is", rggg_num / real_num_files)
print("rggr_num has ", rggr_num, "the rate is", rggr_num / real_num_files)
print("rgrg_num has ", rgrg_num, "the rate is", rgrg_num / real_num_files)
print("rgrr_num has ", rgrr_num, "the rate is", rgrr_num / real_num_files)

print("rrgg_num has ", rrgg_num, "the rate is", rrgg_num / real_num_files)
print("rrgr_num has ", rrgr_num, "the rate is", rrgr_num / real_num_files)
print("rrrg_num has ", rrrg_num, "the rate is", rrrg_num / real_num_files)
print("rrrr_num has ", rrrr_num, "the rate is", rrrr_num / real_num_files)
