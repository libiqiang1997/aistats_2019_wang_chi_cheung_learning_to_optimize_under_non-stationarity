import os
import numpy as np
import multiprocessing as mp
path_strs = os.getcwd().split('\\')
current_dir_name = path_strs[len(path_strs) - 1]
jupyter_notebook = False
if current_dir_name == 'jupyter_notebook':
    jupyter_notebook = True
    os.chdir("..")
from class_Environment import Environment
from class_Simulator import Simulator
from algorithm_class.class_SW_UCB import SW_UCB
from util import plot_regret, modify_regret_figure, plot_supplementary_figure
from datetime import datetime

debug = False
if debug:
    num_mc = 1
    option = 'solo_processing'
else:
    num_mc = 100
    option = 'multi_processing'
time_horizon = 240000
# time_horizon = 240
t_saved = [i for i in range(time_horizon)]
k = 2
d = 2
variation_budget = 1
sigma_noise = 0.1

w = int(d ** (2 / 3) * time_horizon ** (2 / 3) * variation_budget ** (- 2 / 3))
action_norm_bound = 1
theta_norm_bound = 1


def run_experiment(option, num_mc, time_horizon, t_saved, k, d, variation_budget, sigma_noise, policies, num_thread=1):
    bandit_env = Environment(k, d, sigma_noise, variation_budget, time_horizon)
    simulator = Simulator(bandit_env, policies)
    if option == 'solo_processing':
        avg_regret = simulator.run(num_mc, time_horizon, t_saved)
        return avg_regret
    elif option == 'multi_processing':
        avg_regret = simulator.multiprocessing_run(num_thread, num_mc, time_horizon, t_saved)
        return avg_regret


def para_multiprocessing_run(delta, lambda_):
    # delta = 0.01
    # lambda_ = 1
    figure_name = ('swucb_delta' + str(delta) + '_lambda' + str(lambda_)).replace('.', 'dot')
    policies = [SW_UCB(d, w, k, sigma_noise, delta, lambda_, action_norm_bound, theta_norm_bound)]
    line_name = policies[0].__str__()
    avg_regret = run_experiment(option, num_mc, time_horizon, t_saved, k, d, variation_budget, sigma_noise, policies)
    plot_regret(figure_name, line_name, avg_regret, t_saved, jupyter_notebook)


if __name__ == '__main__':
    start_time = datetime.now()
    print('start_time:', start_time)

    if debug:
        # solo_processing
        delta = 0.01
        lambda_ = 1
        figure_name = ('swucb_delta' + str(delta) + '_lambda' + str(lambda_)).replace('.', 'dot')
        policies = [SW_UCB(d, w, k, sigma_noise, delta, lambda_, action_norm_bound, theta_norm_bound)]
        line_name = policies[0].__str__()
        avg_regret = run_experiment(option, num_mc, time_horizon, t_saved, k, d, variation_budget, sigma_noise, policies)
        plot_regret(figure_name, line_name, avg_regret, t_saved, jupyter_notebook)
    else:
        # # multi_processing
        # option = 'multi_processing'
        # num_thread = 10
        # delta = 0.01
        # lambda_ = 1
        # figure_name = ('swucb_delta' + str(delta) + '_lambda' + str(lambda_)).replace('.', 'dot')
        # policies = [SW_UCB(d, w, k, sigma_noise, delta, lambda_, action_norm_bound, theta_norm_bound)]
        # line_name = policies[0].__str__()
        # avg_regret = run_experiment(option, num_mc, time_horizon, t_saved, k, d, variation_budget, sigma_noise, policies)
        # plot_regret(figure_name, line_name, avg_regret, t_saved, jupyter_notebook)

        # multi_processing
        # deltas = [0.01, 0.05, 0.1]
        # lambda_s = [0.1, 0.5, 1]
        deltas = [0.001, 0.005, 0.01]
        lambda_s = [1, 5, 10]
        num_para_thread = len(deltas) * len(lambda_s)
        threads = []
        for i in range(len(deltas)):
            for j in range(len(lambda_s)):
                threads.append(mp.Process(target=para_multiprocessing_run,
                                          args=(deltas[i], lambda_s[j])))
                threads[i * len(lambda_s) + j].start()
        for i in range(num_para_thread):
            threads[i].join()

    end_time = datetime.now()
    cost_time = end_time - start_time
    print('start_time:', start_time)
    print('end_time:', end_time)
    print('cost_time:', cost_time)