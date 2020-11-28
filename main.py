import os
import numpy as np
import multiprocessing as mp

from algorithm_class.class_LinUCB import LinUCB
from algorithm_class.class_SW_UCB import SW_UCB
from algorithm_class.class_BOB import BOB

path_strs = os.getcwd().split('\\')
current_dir_name = path_strs[len(path_strs) - 1]
jupyter_notebook = False
if current_dir_name == 'jupyter_notebook':
    jupyter_notebook = True
    os.chdir("..")
from class_Environment import Environment
from class_Simulator import Simulator
from util import plot_regret, plot_expected_rewards, count_change_points
from datetime import datetime

# debug = True
debug = False
# tune_parameters = True
tune_parameters = False
if debug:
    # time_horizon = 240
    # t_saved_loglog = [(i + 1) * 30 - 1 for i in range(8)]
    time_horizon = 5
    t_saved_loglog = [i for i in range(4)]
else:
    time_horizon = 240000
    t_saved_loglog = [(i + 1) * 30000 - 1 for i in range(8)]


num_thread = 1
num_mc = 1

# parameters
k = 2
d = 2
delta = 0.01
lambda_ = 1
stationary_degree = 1
# variation_budget = 1
variation_budget = time_horizon ** (1/3)
sigma_noise = 0.1
# w = int(d ** (2 / 3) * time_horizon ** (2 / 3) * variation_budget ** (- 2 / 3))   # if variation_budget is known
w = int((d * time_horizon) ** (2 / 3))  # if variation_budget is unknown
# w = time_horizon
action_norm_bound = 1
theta_norm_bound = 1

def run_fixed_parameter():
    figure_name = ('regret_figure')
    bandit_env = Environment(k, d, sigma_noise, variation_budget, time_horizon, stationary_degree)
    # policies = [BOB('BOB', d, sigma_noise, lambda_, action_norm_bound, theta_norm_bound, time_horizon, 'black')]
    # policies = [SW_UCB('SW-UCB', d, w, sigma_noise, delta, lambda_, action_norm_bound, theta_norm_bound, 'red')]
    # policies = [LinUCB('LinUCB', d, sigma_noise, delta, lambda_, action_norm_bound, theta_norm_bound, 'blue')]
    # policies = [LinUCB('OR-LinUCB', d, sigma_noise, delta, lambda_, action_norm_bound, theta_norm_bound, 'green')]
    policies = [BOB('BOB', d, sigma_noise, lambda_, action_norm_bound, theta_norm_bound, time_horizon, 'black'),
                SW_UCB('SW-UCB', d, w, sigma_noise, delta, lambda_, action_norm_bound, theta_norm_bound, 'red')]
    simulator = Simulator(bandit_env, policies, time_horizon)
    regret_dict, expected_rewards_dict = simulator.run(num_thread, num_mc)
    plot_regret(figure_name, policies, regret_dict, t_saved_loglog, jupyter_notebook)
    plot_expected_rewards(expected_rewards_dict, stationary_degree, jupyter_notebook)


def run_different_parameters(thread_id, stationary_degree):
    # delta = 0.01
    # lambda_ = 1
    # figure_name = ('stationary_degrees=5_w=T')
    figure_name = ('stationary_degree' + str(stationary_degree) + '_w' + str(w)).replace('.', 'dot')
    # figure_name = (str(datetime.now().hour).zfill(2) + str(datetime.now().minute).zfill(2) + str(datetime.now().second).zfill(2) \
    #                + '_stationary_degree' + str(stationary_degree)).replace('.', 'dot')
    bandit_env = Environment(k, d, sigma_noise, variation_budget, time_horizon, stationary_degree)
    # policies = [BOB('BOB', d, sigma_noise, lambda_, action_norm_bound, theta_norm_bound, time_horizon, 'black')]
    # policies = [SW_UCB('SW-UCB', d, w, sigma_noise, delta, lambda_, action_norm_bound, theta_norm_bound, 'red')]
    # policies = [LinUCB('LinUCB', d, sigma_noise, delta, lambda_, action_norm_bound, theta_norm_bound, 'blue')]
    # policies = [LinUCB('OR-LinUCB', d, sigma_noise, delta, lambda_, action_norm_bound, theta_norm_bound, 'green')]
    policies = [BOB('BOB', d, sigma_noise, lambda_, action_norm_bound, theta_norm_bound, time_horizon, 'black'),
                SW_UCB('SW-UCB', d, w, sigma_noise, delta, lambda_, action_norm_bound, theta_norm_bound, 'red')]
    simulator = Simulator(bandit_env, policies, time_horizon)
    regret_dict, expected_rewards_dict = simulator.run(num_thread, num_mc)
    plot_regret(figure_name, policies, regret_dict, t_saved_loglog, jupyter_notebook)
    plot_expected_rewards(expected_rewards_dict, stationary_degree, jupyter_notebook)


if __name__ == '__main__':
    start_time = datetime.now()
    print('start_time:', start_time)

    threads = []
    if not tune_parameters:
        threads.append(mp.Process(target=run_fixed_parameter))
        threads[0].start()
        threads[0].join()
    else:
        stationary_degrees = [3, 1, 0.5, 0.4, 0.3, 0.2, 0.1]
        for i in range(len(stationary_degrees)):
            thread_id = i
            threads.append(mp.Process(target=run_different_parameters,
                                      args=(thread_id, stationary_degrees[i])))
            threads[i].start()
        for thread in threads:
            thread.join()

    end_time = datetime.now()
    cost_time = end_time - start_time
    print('start_time:', start_time)
    print('end_time:', end_time)
    print('cost_time:', cost_time)