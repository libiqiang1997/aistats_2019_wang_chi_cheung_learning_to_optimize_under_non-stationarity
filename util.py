from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import loadmat
from scipy.io import savemat


# plt parameters
colors = ['black', 'red', 'green', 'blue']
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.linestyle'] = '--'
plt.rcParams['legend.fontsize'] = 27
plt.rcParams['axes.labelsize'] = 27
# plt.rcParams['axes.labelweight'] = 'bold'
# ax = plt.gca()
# ax.xaxis.set_tick_params(labelsize=20)
plt.rcParams['xtick.labelsize'] = 27
plt.rcParams['ytick.labelsize'] = 27
plt.rcParams['figure.subplot.left'] = 0.225
plt.rcParams['figure.subplot.bottom'] = 0.17
plt.rcParams['figure.subplot.right'] = 0.96
plt.rcParams['figure.subplot.top'] = 0.99
# plt.rcParams['text.usetex'] = True
# print(plt.rcParams)

# create dir
current_path = os.getcwd()
png_path = 'output/png'
png_save_path = current_path + '/' + png_path
if not os.path.exists(png_save_path):
    os.makedirs(png_save_path)
eps_path = 'output/eps'
eps_save_path = current_path + '/' + eps_path
if not os.path.exists(eps_save_path):
    os.makedirs(eps_save_path)
mat_path = 'output/mat'
mat_save_path = current_path + '/' + mat_path
if not os.path.exists(mat_save_path):
    os.makedirs(mat_save_path)


def plot_regret(figure_name, policies, avg_regret_dict, t_saved_loglog, jupyter_notebook):
    plot_figure(policies, avg_regret_dict)
    if not jupyter_notebook:
        save_figure(figure_name)
        save_data(figure_name, policies, avg_regret_dict)

    figure_name_loglog = figure_name + '_loglog'
    avg_regret_loglog_dict = {}
    for policy in policies:
        avg_regret_loglog_dict[policy.name] = avg_regret_dict[policy.name][t_saved_loglog]
    plot_loglog_figure(policies, avg_regret_loglog_dict, t_saved_loglog)
    if not jupyter_notebook:
        save_figure(figure_name_loglog)
        save_data(figure_name_loglog, policies, avg_regret_loglog_dict)


# def modify_regret_figure(figure_name, data_name, jupyter_notebook):
#     average_regret = load_data(figure_name, policies, data_name)
#     plot_loglog_figure(data_name, average_regret)
#     if not jupyter_notebook:
#         save_figure(figure_name)


def plot_figure(policies, avg_regret_dict):
    fig = plt.figure() # default (6.4, 4.8) 640x480
    plt.xlabel(r'Round $t$')
    plt.ylabel(r'Cumulative Regret $R_t$')
    for policy in policies:
        plt.plot(range(1, len(avg_regret_dict[policy.name]) + 1), avg_regret_dict[policy.name], label=policy.name, color=policy.color)
        plt.legend()

    # plt.plot(range(1, len(avg_regret_dict) + 1), avg_regret_dict, label=policies, color=colors[3])
    # plt.xscale('log', nonpositive='clip')
    # plt.yscale('log', nonpositive='clip')
    # for i in range(len(t_saved)):
    #     plt.scatter(t_saved[i], line_regret[i], color=colors[3])
    # plt.xlim((1, 10**6))
    # plt.xlim((10**3, t_saved[len(t_saved) - 1] * 1.2))
    # plt.ylim((10 ** 3, line_regret[len(t_saved) - 1] * 1.2))


def plot_loglog_figure(policies, avg_regret_loglog_dict, t_saved_loglog):
    fig = plt.figure() # default (6.4, 4.8) 640x480
    plt.xlabel(r'Round $t$')
    plt.ylabel(r'Cumulative Regret $R_t$')
    for policy in policies:
        avg_regret_loglog = avg_regret_loglog_dict[policy.name]
        plt.plot(t_saved_loglog, avg_regret_loglog, label=policy.name, color=policy.color)
    plt.xscale('log', nonpositive='clip')
    plt.yscale('log', nonpositive='clip')
    # plt.xlim((1, t_saved[len(t_saved) - 1]))
    for policy in policies:
        for i in range(len(t_saved_loglog)):
            plt.scatter(t_saved_loglog[i], avg_regret_loglog_dict[policy.name][i], color=policy.color)
    # plt.xlim((1, 10**6))
    plt.xlim((10 ** 2, t_saved_loglog[len(t_saved_loglog) - 1] * 1.2))
    plt.ylim((1, 5 * 10 ** 5))
    plt.legend()


def save_figure(figure_name):
    plt.savefig(png_save_path+ '/' + figure_name)
    plt.savefig(eps_save_path + '/' + figure_name + '.eps', format='eps')


def save_data(figure_name, policies, avg_regret_dict):
    data_dict = {}
    for policy in policies:
        data_dict[policy.name] = avg_regret_dict[policy.name]
    # data_dict = {policies: avg_regret_dict}
    savemat(mat_save_path + '/' + figure_name + '.mat', data_dict)


# def load_data(figure_name, policies, data_name):
#     data_dict = loadmat(mat_save_path + '/' + figure_name)
#     avg_regret_dict = {}
#     for policy in policies:
#         data_dict[policy.name] = avg_regret_dict[policy.name]
#     average_regret = data_dict[data_name][0]
#     return average_regret


def plot_expected_rewards(expected_rewards_dict, stationary_degree, jupyter_notebook):
    expected_rewards1 = expected_rewards_dict[1]
    expected_rewards2 = expected_rewards_dict[2]
    fig = plt.figure()
    time_horizon = len(expected_rewards1)
    time_axies = range(1, time_horizon + 1)
    plt.plot(time_axies, expected_rewards1, label=r'$\mu_{1,t}$', color=colors[3])
    plt.plot(time_axies, expected_rewards2, label=r'$\mu_{2,t}$', color=colors[1])
    plt.xlabel(r'Round $t$')
    plt.ylabel(r'Expected Reward $\mu_t$')
    plt.legend()
    # figure_name = (str(datetime.now().hour).zfill(2) + str(datetime.now().minute).zfill(2) + str(datetime.now().second).zfill(2) \
    #               + '_expected_rewards' + '_stationary_degree' + str(stationary_degree)).replace('.', 'dot')
    figure_name = ('stationary_degree' + str(stationary_degree) + '_expected_rewards').replace('.', 'dot')
    if not jupyter_notebook:
        save_figure(figure_name)


def count_change_points(stationary_degree, expected_rewards_dict):
    count = 0
    for i in range(len(expected_rewards_dict[1])):
        if np.isclose(expected_rewards_dict[1][i], expected_rewards_dict[2][i]):
            count += 1
    print(stationary_degree, count)

# plot_supplementary_figure()
# # modify_figure
# figure_name = 'ucb_convergence_verification'
# line_name = 'UCB'
# modify_regret_figure(figure_name, line_name)