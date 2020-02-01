# This is a script to run all the tests. Can be helpful to run before pushing your code.

import subprocess
import os
import time


def run_all_tests():
    # Suppress the tensorflow logs so test output is easier to read
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    files_to_test = [os.path.join(os.path.abspath(os.getcwd()), 'tests', f) for f in os.listdir('./tests')
                     if f.startswith('test')]

    for file in files_to_test:
        exit_code = run_test(file)
        if exit_code != 0:
            print('Test failed: %s' % os.path.basename(file))
            break


def run_test(filename):
    """
    Run a particular test defined by the filename.
    """
    print('======================================================================')
    print('Running test %s' % os.path.basename(filename))
    t_start = time.time()
    exit_code = subprocess.call(["/home/anjianl/Desktop/application/anaconda3/envs/venv-mpc/bin/python3", filename])
    t_end = time.time()
    print('Execution time for test %s is %d seconds.' % (os.path.basename(filename), t_end - t_start))
    print('======================================================================')
    return exit_code

if __name__ == '__main__':
    run_all_tests()
