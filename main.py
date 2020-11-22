import os
import numpy as np
import multiprocessing as mp

from algorithm_class.class_LinUCB import LinUCB

path_strs = os.getcwd().split('\\')
current_dir_name = path_strs[len(path_strs) - 1]
jupyter_notebook = False
if current_dir_name == 'jupyter_notebook':
    jupyter_notebook = True
    os.chdir("..")
from class_Environment import Environment
from class_Simulator import Simulator
from algorithm_class.class_SW_UCB import SW_UCB
from util import plot_regret, plot_expected_rewards
from datetime import datetime

debug = True
# debug = False
if debug:
    # debug setting for small horizon and 1 thread
    num_thread = 1
    num_mc = 1
    # time_horizon = 240
    # t_saved_loglog = [(i + 1) * 30 - 1 for i in range(8)]
    time_horizon = 240000
    t_saved_loglog = [(i + 1) * 30000 - 1 for i in range(8)]
else:
    num_thread = 1
    num_mc = 1
    time_horizon = 240000
    t_saved_loglog = [(i + 1) * 30000 - 1 for i in range(8)]
# t_saved = [i for i in range(time_horizon)]

# parameters
k = 2
d = 2
delta = 0.01
# lambda_ = 1
variation_budget = 1
sigma_noise = 1
w = int(d ** (2 / 3) * time_horizon ** (2 / 3) * variation_budget ** (- 2 / 3))
action_norm_bound = 1
theta_norm_bound = 1


def run_experiment(num_mc, time_horizon, k, d, variation_budget, sigma_noise, policies):
    bandit_env = Environment(k, d, sigma_noise, variation_budget, time_horizon)
    simulator = Simulator(bandit_env, policies)
    # avg_regret = simulator.multiprocessing_run(num_thread, num_mc, time_horizon)
    avg_regret_dict = simulator.multiprocessing_run(num_thread, num_mc, time_horizon)
    return avg_regret_dict


def para_multiprocessing_run(lambda_, w):
    # delta = 0.01
    # lambda_ = 1
    # figure_name = ('swucb_lambda' + str(lambda_) + '_w' + str(w)).replace('.', 'dot')
    figure_name = ('non_stationary_bandit')
    # figure_name = ('swucb_w' + str(w)).replace('.', 'dot')
    # policies = [SW_UCB('SW-UCB', d, w, sigma_noise, delta, lambda_, action_norm_bound, theta_norm_bound, 'red')]
    # policies = [LinUCB('LinUCB', d, sigma_noise, delta, lambda_, action_norm_bound, theta_norm_bound, 'blue')]
    # policies = [LinUCB('OR-LinUCB', d, sigma_noise, delta, lambda_, action_norm_bound, theta_norm_bound, 'green')]
    # policies = [SW_UCB('SW-UCB', d, w, sigma_noise, delta, lambda_, action_norm_bound, theta_norm_bound, 'red'),
    #             LinUCB('LinUCB', d, sigma_noise, delta, lambda_, action_norm_bound, theta_norm_bound, 'blue'),
    #             LinUCB('OR-LinUCB', d, sigma_noise, delta, lambda_, action_norm_bound, theta_norm_bound, 'green')]
    policies = [SW_UCB('SW-UCB', d, w, sigma_noise, delta, lambda_, action_norm_bound, theta_norm_bound, 'red'),
                LinUCB('OR-LinUCB', d, sigma_noise, delta, lambda_, action_norm_bound, theta_norm_bound, 'green')]
    # line_name = policies[0].__str__()
    avg_regret_dict = run_experiment(num_mc, time_horizon, k, d, variation_budget, sigma_noise, policies)
    plot_regret(figure_name, policies, avg_regret_dict, t_saved_loglog, jupyter_notebook)


if __name__ == '__main__':
    start_time = datetime.now()
    print('start_time:', start_time)

    if debug:
        lambda_s = [0.1]
        ws = [w]
    else:
        ws = [600 + i * 600 for i in range(20)] + [10000 + i * 10000 for i in range(6)]
        ws.append(w)
        ws.sort()
        # print('ws:', ws)
        # print('len(ws):', len(ws))
        lambda_s = [0.01, 0.1, 1]

    num_para_pair = len(ws) * len(lambda_s)
    print('num_para_pair:', num_para_pair)
    threads = []
    for i in range(len(lambda_s)):
        for j in range(len(ws)):
            threads.append(mp.Process(target=para_multiprocessing_run, args=(lambda_s[i], ws[j])))
            threads[i * len(ws) + j].start()
    for i in range(num_para_pair):
        threads[i].join()


    end_time = datetime.now()
    cost_time = end_time - start_time
    print('start_time:', start_time)
    print('end_time:', end_time)
    print('cost_time:', cost_time)