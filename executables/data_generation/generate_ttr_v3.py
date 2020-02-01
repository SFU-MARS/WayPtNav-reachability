import tensorflow as tf
import numpy as np
import argparse
import importlib
import pickle
import os
import time

from utils import utils, log_utils

tf.enable_eager_execution(**utils.tf_session_config())



class GenerateTTR(object):
    """
    Generate and save reach avoid TTR map. Specify "version" for different maps to generate on
    """

    def run(self):

        with tf.device('/cpu:0'):

            simulator = self.get_simulator()

            num = 10000

            for i in range(num):
                start_time = time.time()
                simulator.reset()
                end_time = time.time()
                print("episode", i, "takes", end_time-start_time)


    def get_simulator(self):

        parser = argparse.ArgumentParser(description='Process the command line inputs')
        parser.add_argument("-p", "--params", required=True, help='the path to the parameter file')
        parser.add_argument("-d", "--device", type=int, default=1, help='the device to run the training/test on')
        args = parser.parse_args()

        p = self.create_params(args.params)

        p.simulator_params = p.data_creation.simulator_params
        p.simulator_params.simulator.parse_params(p.simulator_params)

        simulator = p.simulator_params.simulator(p.simulator_params)

        return simulator

    def create_params(self, param_file):
        """
        Create the parameters given the path of the parameter file.
        """
        # Execute this if python > 3.4
        try:
            spec = importlib.util.spec_from_file_location('parameter_loader', param_file)
            foo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(foo)
        except AttributeError:
            # Execute this if python = 2.7 (i.e. when running on real robot with ROS)
            module_name = param_file.replace('/', '.').replace('.py', '')
            foo = importlib.import_module(module_name)
        return foo.create_params()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    GenerateTTR().run()
