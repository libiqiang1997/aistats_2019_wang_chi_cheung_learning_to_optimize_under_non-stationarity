import os
import numpy as np

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


def run_experiment(num_mc, t_saved, option):
    bandit_env = Environment(k, d, sigma_noise, variation_budget, time_horizon)
    simulator = Simulator(bandit_env, policies)
    if option == 'solo_processing':
        avg_regret = simulator.run(num_mc, time_horizon, t_saved)
        return avg_regret
    elif option == 'multi_processing':
        avg_regret = simulator.multiprocessing_run(num_thread, num_mc, time_horizon, t_saved)
        return avg_regret

debug = True
if debug:
    # time_horizon = 2400
    time_horizon = 6
    figure_type = 'xy'
    # t_saved = [(i + 1) * 300 - 1 for i in range(8)]
    t_saved = [i for i in range(time_horizon)]
else:
    time_horizon = 240000
    figure_type = 'logxlogy'
    t_saved = [(i + 1) * 30000 - 1 for i in range(8)]

k = 2
d = 2
w = 3
delta = 0.01
lambda_ = 1
if debug:
    num_mc = 1
else:
    num_mc = 100
variation_budget = 1
line_name = 'UCB'
# number_t_saved = 20
# t_saved = np.int_(np.linspace(0, time_horizon - 1, number_t_saved))
# t_saved = [i for i in range(time_horizon)]
# print(t_saved)
if __name__ == '__main__':
    start_time = datetime.now()
    print('start_time:', start_time)

    if debug:
        # solo_processing
        option = 'solo_processing'
        sigma_noise = 1
        figure_name = ('ucb_convergence_verification_sigma' + str(sigma_noise)).replace('.', 'dot')
        policies = [SW_UCB(d, w, sigma_noise, k, delta, lambda_)]
        avg_regret = run_experiment(num_mc, t_saved, option)
        plot_regret(figure_name, line_name, avg_regret, t_saved, figure_type, jupyter_notebook)
    else:
        # multi_processing
        option = 'multi_processing'
        num_thread = 10
        sigma_noise = 1
        figure_name = ('ucb_convergence_verification_sigma' + str(sigma_noise)).replace('.', 'dot')
        policies = [SW_UCB(d, sigma_noise, k, delta, lambda_)]
        avg_regret = run_experiment(num_mc, t_saved, option)
        plot_regret(figure_name, line_name, avg_regret, t_saved, figure_type, jupyter_notebook)

    # # modify_figure
    # figure_name = 'ucb_convergence_verification_sigma1'
    # modify_regret_figure(figure_name, line_name, jupyter_notebook)

    # plot_supplementary_figure()

    end_time = datetime.now()
    cost_time = end_time - start_time
    print('start_time:', start_time)
    print('end_time:', end_time)
    print('cost_time:', cost_time)