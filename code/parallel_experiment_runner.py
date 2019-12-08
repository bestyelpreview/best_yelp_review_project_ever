"""!
@brief A useful script for running multiple experiment configurations in
parallel

@authors Zhepei Wang <zhepeiw03@gmail.com>
         Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

from multiprocessing import Lock, Process, Queue, current_process
import time
import queue
import os
import sys
import argparse
import copy
sys.path.append(os.path.join(os.path.dirname(__file__)))
import tools.argtools
from main import safe_run_experiment
from main import experiment2string


def get_experiment(experiments_to_run,
                   successful_experiments,
                   failed_experiments,
                   cuda_id):
    while True:
        try:
            experiment_config = experiments_to_run.get_nowait()
            experiment_config.cuda_available_devices = cuda_id
            print('I am going to run the experiment {} on cuda '
                  'device: {}'.format(
                experiment2string(experiment_config), cuda_id))
            print(cuda_id)
            safe_run_experiment(experiment_config)
            successful_experiments.put(experiment2string(experiment_config))
        except queue.Empty:
            break
        except Exception as e:
            failed_experiments.put(
                '{} Failed by raising: {}'.format(
                    experiment2string(experiment_config), e))
            time.sleep(5)

    return True


def parallel_experiments_runner(configs_list, available_gpus):
    experiments_to_run = Queue()
    successful_experiments = Queue()
    failed_experiments = Queue()
    processes = []

    for config_args in configs_list:
        experiments_to_run.put(config_args)

    # creating processes
    for cuda_id in available_gpus:
        p = Process(target=get_experiment,
                    args=(experiments_to_run,
                          successful_experiments,
                          failed_experiments,
                          cuda_id))
        processes.append(p)
        p.start()
        time.sleep(5)

    # completing process
    for p in processes:
        p.join()

    print('Successful Experiments')
    while not successful_experiments.empty():
        print(successful_experiments.get())


    print('Failed Experiments')
    while not failed_experiments.empty():
        print(failed_experiments.get())

    return True



def create_configs(args):
    dic = vars(args)
    configs_list = []
    for m in dic['model_type']:
        for l in dic['num_layers']:
            for h in dic['num_hidden_units']:
                for d in dic['dropout_rate']:
                    for e in dic['embedding_size']:
                        for v in dic['vocab_size']:
                            args_copy = copy.deepcopy(args)
                            args_copy.model_type = m
                            args_copy.num_layers = l
                            args_copy.num_hidden_units = h
                            args_copy.dropout_rate = d
                            args_copy.embedding_size = e
                            args_copy.vocab_size = v
                            configs_list.append(args_copy)

    return configs_list


if __name__ == '__main__':
    # main()
    # configs_list = [4, 5, 6, 7, 8, 9]
    # available_gpus = [0, 1, 3]
    # parallel_experiments_runner(configs_list, available_gpus)
    args = tools.argtools.get_args(parallel_experiments=True)
    configs_list = create_configs(args)
    available_gpus = args.cuda_available_devices

    parallel_experiments_runner(configs_list, available_gpus)