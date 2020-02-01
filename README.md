WayPtNav-reachability
==========
Welcome to WayPtNav-reachability, a codebase for generating robust supervision for learning-based robot visual navigation using Hamilton-Jacobi Reachability (HJ). We are a team of researchers from Simon Fraser University and UC Berkeley.

In this codebase we explore ["Generating Robust Supervision for Learning-Based Visual Navigation Using Hamilton-Jacobi Reachability"](https://anjianli21.github.io/research/reachability-based/). We provide code to generate training data through solving HJ Partial Differential Equation, to train your own agent, and to deploy it in a variety of different simulations rendered from scans of Stanford Buildings.

This code is written mainly in Python, with MATLAB and C++ needed for reachability computation.

More code information on WayPtNav, please see [WayPtNav_README.md](WayPtNav-README.md)

## Setup
### Setup A Virtual Environment
```
conda create -n WayPtNav-reachability python=3.6
source activate WayPtNav-reachability
pip install -U pip
```

### Install modules
All necessary modules are included in [requirement.txt](requirements.txt). We recommend to follow the instructions in [module_installation.txt](module_installation.txt) to install for version compatibility.

#### Patch the OpenGL Installation
In the terminal from the root directory of the project run the following commands.
```
1. cd mp_env
2. bash patches/apply_patches_3.sh
```
If the script fails there are instructions in apply_patches_3.sh describing how to manually apply the patch.

### Download the necessary data (~13 GB)
```
TODO: Put Something Here
```
#### (Optional). Download the training data used in training the model-based and end-to-end methods. (~82 GB)
```
TODO: Put Something Here
```

### Configure the data dir
##### Configure the Control Pipeline Data Directory
In ./params/control_pipeline_params.py change the following line
```
p.dir = 'PATH/TO/DATA/control_pipelines' 
```
##### Configure the Reachability Data Directory
In ./params/reachability_map/reachability_map_params.py change the following line
```
p.MATLAB_PATH = 'PATH/TO/PROJECT/reachability'
```
MATLAB_PATH should be set to ./reachability to project folder.
#####
##### Configure the Stanford Building Parser Dataset Data Directory
In ./params/renderer_params.py change the following line
```
def get_sbpd_data_dir():
	return 'PATH/TO/DATA/stanford_building_parser_dataset'
```
##### Configure the Pretrained Weights Data Directory
In ./params/rgb_trainer/sbpd/projected_grid/resnet50/rgb_waypoint_trainer_finetune_params.py change the following line:
```
p.trainer.ckpt_path = 'PATH/TO/DATA/pretrained_weights/WayPtNav/session_2019-01-27_23-32-01/checkpoints/ckpt-9'
```
In ./params/rgb_trainer/sbpd/projected_grid/resnet50/rgb_control_trainer_finetune_params.py change the following line:
```
p.trainer.ckpt_path = 'PATH/TO/DATA/pretrained_weights/E2E/session_2019-01-27_23-34-22/checkpoints/ckpt-18'
```
##### Configure the Expert Success Goals Data Directory
In ./params/rgb_trainer/sbpd/projected_grid/resnet50/rgb_waypoint_trainer_finetune_params.py and ./params/rgb_trainer/sbpd/projected_grid/resnet50/rgb_control_trainer_finetune_params.py change the following line:
```
p.test.expert_success_goals = DotMap(use=True,
							         dirname='PATH/TO/DATA/expert_success_goals/sbpd_projected_grid')
```
##### Configure the Training Data Directory (Optional)
In ./params/rgb_trainer/sbpd/projected_grid/resnet50/rgb_waypoint_trainer_finetune_params.py and ./params/rgb_trainer/sbpd/projected_grid/resnet50/rgb_control_trainer_finetune_params.py change the following line:
```
p.data_creation.data_dir = ['PATH/TO/DATA/sbpd_projected_grid_include_last_step_successful_goals_only/area3/full_episode_random_v1_100k',
						    'PATH/TO/DATA/sbpd_projected_grid_include_last_step_successful_goals_only/area4/full_episode_random_v1_100k',
   							'PATH/TO/DATA/sbpd_projected_grid_include_last_step_successful_goals_only/area5a/full_episode_random_v1_100k']
```
### Run the Tests
First, comment the following code in ./objectives/objective_function.py line 18. (WARNING: remember to uncomment after running the test!)
```
from objectives.goal_distance import GoalDistance
```
Then, to ensure you have successfully installed the WayptNav codebase run the following command. All tests should pass.
```
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python PATH/TO/PROJECT/executables/run_all_tests.py
```
## Getting Started
#### Overview
The WayptNav codebase is designed to allow you to:

	1. Create training data using an expert policy based on HJ Reachability
	2. Train a network (for either model based or end-to-end navigation)  
	3. Test a trained network

Each of these 3 tasks can be run via an executable file. All of the relevant executable files are located in the ./executables subdirectory of the main project. To use an executable file the user must specify
```
1. mode (generate_data, train, or test)
2. job_dir (where to save the all relevant data from this run of the executable)
3. params (which parameter file to use)
4. device (which device to run tensorflow on. -1 forces CPU, 0+ force the program to run on the corresponding GPU device)
```
When creating training data, computing reachability information (Time-to-reach value, TTR) takes comparatively more time. Thus we provide a parallel method to generate TTR for 3 area (area3, area4, area5a) simultaneously and save the precomputed TTR for future use. Relevant executable files are located in the ./executables/data_generation subdirectory of the main project.

We also provide some helper functions in ./executables/helper subdirectory of the main project, including computing trajectory metrics, plotting metrics etc.

#### A simple example: Generate Data, Train, and Test a Sine Function
We have provided a simple example to train a sine function for your understanding. To generate data, train and test the sine function example using GPU 0 run the following 3 commands:
```
1. PYTHONPATH='.' python PATH/TO/PROJECT/executables/sine_function_trainer generate-data --job-dir JOB_DIRECTORY_NAME_HERE --params params/sine_params.py -d 0
2. PYTHONPATH='.' python PATH/TO/PROJECT/executables/sine_function_trainer train --job-dir JOB_DIRECTORY_NAME_HERE --params params/sine_params.py -d 0

In ./params/sine_params.py change p.trainer.ckpt_path to point to a checkpoint from the previously run training session. For example:

3a. p.trainer.ckpt_path = 'PATH/TO/PREVIOUSLY/RUN/SESSION/checkpoints/ckpt-10'

3b. PYTHONPATH='.' python PATH/TO/PROJECT/executables/sine_function_trainer test --job-dir JOB_DIRECTORY_NAME_HERE --params params/sine_params.py -d 0
```

The output of testing the sine function will be saved in 'PATH/TO/PREVIOUSLY/RUN/SESSION/TEST/ckpt-10'.

## Generating Training Data with HJ Reachability

Before we start, please make sure you successfully installed MATLAB and configured the Python-MATLAB interface. See [here](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html) for more details.

In this section, we compute and save TTR and TTC value maps in the ./reachability/data_tmp subdirectory and use the value maps to generate training data. If the data is precomputed, we directly load them for data generation.

#### Change the data_dir to reflect the desired directory for your new data
In params/rgb_trainer/sbpd/projected_grid/resnet50/rgb_waypoint_trainer_finetune_params.py
```
p.data_creation.data_dir = 'PATH/TO/NEW/DATA'
```
Run the following command to create new data.
```
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python PATH/TO/PROJECT/executables/rgb/resnet50/rgb_waypoint_trainer.py generate-data --job-dir PATH/TO/LOG/DIR --params params/rgb_trainer/sbpd/projected_grid/resnet50/rgb_waypoint_trainer_finetune_params.py -d 0
```

#### Parallel computation
To compute TTR in 3 areas simultaneously, run the following commands in parallel, which comsumes ~10GB GPU memory.
```
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python PATH/TO/PROJECT/executables/data_generation/generate_ttr.py generate-data --job-dir PATH/TO/LOG/DIR --params params/rgb_trainer/sbpd/projected_grid/resnet50/rgb_waypoint_trainer_finetune_params.py -d -1
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python PATH/TO/PROJECT/executables/data_generation/generate_ttr_v2.py generate-data --job-dir PATH/TO/LOG/DIR --params params/rgb_trainer/sbpd/projected_grid/resnet50/rgb_waypoint_trainer_finetune_params_v2.py -d -1
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python PATH/TO/PROJECT/executables/data_generation/generate_ttr_v3.py generate-data --job-dir PATH/TO/LOG/DIR --params params/rgb_trainer/sbpd/projected_grid/resnet50/rgb_waypoint_trainer_finetune_params_v3.py -d -1
```
The computed TTR and TTC will save in ./reachability/data_tmp subdirectory.

## Training Your Own Networks
We provide the training data we used to train both the WayPtNav and end-to-end methods. You can experiment with training your own models on our training data or your own generated data using the following commands:
### Train with WayPtNav Method
```
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python PATH/TO/PROJECT/executables/rgb/resnet50/rgb_waypoint_trainer.py train --job-dir PATH/TO/LOG/DIR --params params/rgb_trainer/sbpd/projected_grid/resnet50/rgb_waypoint_trainer_finetune_params.py -d 0
```
### Train A Comparable End-to-End Method
```
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python PATH/TO/PROJECT/executables/rgb/resnet50/rgb_control_trainer.py train --job-dir PATH/TO/LOG/DIR --params params/rgb_trainer/sbpd/projected_grid/resnet50/rgb_control_trainer_finetune_params.py -d 0
```
## Testing Networks
### Test Our WayPtNav-Based Method
Example Command
```
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python PATH/TO/PROJECT/executables/rgb/resnet50/rgb_waypoint_trainer.py test --job-dir reproduce_WayptNavResults --params params/rgb_trainer/sbpd/projected_grid/resnet50/rgb_waypoint_trainer_finetune_params.py -d 0
```

### Test A Comparable End-to-End Method
Example Command
```
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python PATH/TO/PROJECT/executables/rgb/resnet50/rgb_control_trainer.py test --job-dir reproduce_E2EResults --params params/rgb_trainer/sbpd/projected_grid/resnet50/rgb_control_trainer_finetune_params.py -d 0
```

## Citing This Work
If you use the WayPtNav simulator or algorithms in your research please cite:
```
@article{li2019generating,
  title={Generating Robust Supervision for Learning-Based Visual Navigation Using Hamilton-Jacobi Reachability},
  author={Li, Anjian and Bansal, Somil and Giovanis, Georgios and Tolani, Varun and Tomlin, Claire and Chen, Mo},
  journal={arXiv preprint arXiv:1912.10120},
  year={2019}
}
```
