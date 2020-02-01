from dotmap import DotMap
from params.reachability_map.reachability_map_params import create_reachability_data_dir_params


def create_rgb_trainer_params():
    from params.simulator.sbpd_simulator_params import create_params as create_simulator_params
    from params.visual_navigation_trainer_params import create_params as create_trainer_params

    from params.waypoint_grid.sbpd_image_space_grid import create_params as create_waypoint_params
    from params.model.resnet50_arch_v1_params import create_params as create_model_params

    from params.reachability_map.reachability_map_params import create_params as create_reachability_map_params

    # Load the dependencies
    simulator_params = create_simulator_params()

    # Seed
    simulator_params.seed = 10

    # Ensure the waypoint grid is projected SBPD Grid
    simulator_params.planner_params.control_pipeline_params.waypoint_params = create_waypoint_params()

    # Ensure the renderer modality is rgb
    simulator_params.obstacle_map_params.renderer_params.camera_params.modalities = ['rgb']
    simulator_params.obstacle_map_params.renderer_params.camera_params.img_channels = 3
    simulator_params.obstacle_map_params.renderer_params.camera_params.width = 1024
    simulator_params.obstacle_map_params.renderer_params.camera_params.height = 1024
    simulator_params.obstacle_map_params.renderer_params.camera_params.im_resize = 0.21875

    # Change episode horizon
    simulator_params.episode_horizon_s = 80.0

    # Ensure the renderer is using area3
    # TODO: When generating our own data, choose area 3, area4, area5
    #  When testing, choosing area 1
    simulator_params.obstacle_map_params.renderer_params.building_name = 'area3'
    # TODO: area3: thread='v1'; area4: thread='v2'; area5a: thread='v3'
    simulator_params.reachability_map_params.thread = 'v1'
    # specify reachability data dir name according to building_name and thread
    create_reachability_data_dir_params(simulator_params.reachability_map_params)

    # Save trajectory data
    simulator_params.save_trajectory_data = True

    # Choose cost function
    # TODO: in training, always use 'heuristics'
    # simulator_params.cost = 'heuristics'  # heuristics cost.
    simulator_params.cost = 'reachability'

    p = create_trainer_params(simulator_params=simulator_params)

    # Create the model params
    p.model = create_model_params()

    return p


def create_params():
    p = create_rgb_trainer_params()

    # Change the number of inputs to the model
    p.model.num_outputs = 3  # (x, y ,theta)

    # Image size to [224, 224, 3]
    p.model.num_inputs.image_size = [224, 224, 3]

    # Finetune the resnet weights
    p.model.arch.finetune_resnet_weights = True

    # Change the learning rate and num_samples
    p.trainer.lr = 1e-4

    p.trainer.batch_size = 48

    # Todo: num_samples are too large
    p.trainer.num_samples = int(150e3)

    # Checkpoint settings
    p.trainer.ckpt_save_frequency = 1
    p.trainer.restore_from_ckpt = False
    p.trainer.num_epochs = 10

    # Change the Data Processing parameters
    p.data_processing.input_processing_function = 'resnet50_keras_preprocessing_and_distortion'

    # Input processing parameters
    p.data_processing.input_processing_params = DotMap(
        p=0.1,  # Probability of distortion
        version='v1'  # For new FOV (Intel Realsense D435, use version='v3')
    )

    # Checkpoint directory (choose which checkpoint to test)
    # oldcost WayPtNav
    # p.trainer.ckpt_path = '/home/anjianl/Desktop/project/WayPtNav/data/pretrained_weights/WayPtNav/session_2019-01-27_23-32-01/checkpoints/ckpt-9'
    # reachability WayPtNav
    p.trainer.ckpt_path = '/home/anjianl/Desktop/project/WayPtNav/log/train_data_rgb_waypoint/session_2019-09-10_23-56-43_lr-4_reg-6_v2/checkpoints/ckpt-19'

    # Change the data_dir
    # TODO: data dir name is a hack. Allowable name is xxx/area3/xxxx. The second last name
    #  should be the building name
    # TODO: In training, we have to render images for each area individually. That is,
    #   we will have only one area in data_dir each time, run training script, until the metadata is generated. After
    #   finishing this, we uncomment all data directories of all areas and start training together
    # reachability data (with disturbances)
    # p.data_creation.data_dir = [
    #     '/home/anjianl/Desktop/project/WayPtNav/data/successful_data/v2_no_filter_obstacle/area3/success_v2_50k',
    #     '/home/anjianl/Desktop/project/WayPtNav/data/successful_data/v2_no_filter_obstacle/area4/success_v2_13k',
    #     '/home/anjianl/Desktop/project/WayPtNav/data/successful_data/v2_no_filter_obstacle/area4/success_v2_23k',
    #     '/home/anjianl/Desktop/project/WayPtNav/data/successful_data/v2_no_filter_obstacle/area4/success_v2_56k',
    #     '/home/anjianl/Desktop/project/WayPtNav/data/successful_data/v2_no_filter_obstacle/area5a/success_v2_8k',
    #     '/home/anjianl/Desktop/project/WayPtNav/data/successful_data/v2_no_filter_obstacle/area5a/success_v2_44k',
    #     '/home/anjianl/Desktop/project/WayPtNav/data/successful_data/v2_no_filter_obstacle/area5a/success_v2_45']
    # tmp test
    p.data_creation.data_dir = ['/home/anjianl/Desktop/project/WayPtNav/data/tmp/area3/tmp']

    p.data_creation.data_points = int(100e3)
    p.data_creation.data_points_per_file = int(1e3)

    # Seed for selecting the test scenarios and the number of such scenarios
    p.test.seed = 10
    p.test.number_tests = 200

    # Test the network only on goals where the expert succeeded
    p.test.expert_success_goals = DotMap(use=False,
                                         dirname='/home/anjianl/Desktop/project/WayPtNav/data/expert_success_goals/sbpd_projected_grid')

    # Let's not look at the expert
    p.test.simulate_expert = False

    # Parameters for the metric curves
    p.test.metric_curves = DotMap(start_ckpt=1,
                                  end_ckpt=10,
                                  start_seed=1,
                                  end_seed=10,
                                  plot_curves=True
                                  )

    return p
