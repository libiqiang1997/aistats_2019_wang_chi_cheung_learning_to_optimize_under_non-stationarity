import numpy as np
from tqdm import tqdm
import multiprocessing as mp


class Simulator(object):
    def __init__(self, env, policies):
        self.env = env
        self.policies = policies

    # def multiprocessing_run_each_thread(self, thread_id, thread_num_mc, time_horizon, t_saved, thread_avg_regret_dict):
    #     optimal_expected_rewards = np.zeros(time_horizon)
    #     selected_expected_rewards = np.zeros(time_horizon)
    #     n_sub = len(t_saved)
    #     cum_regret = np.zeros((thread_num_mc, n_sub))
    #     for n_experiment in range(thread_num_mc):
    #         print('thread_id, n_experiment:', thread_id, ',', n_experiment)
    #         self.env.re_init()
    #         policy = self.policies[0]
    #         policy.re_init()
    #         for t in range(time_horizon):
    #             self.env.update()
    #             choice = policy.select_arm(self.env.arms)
    #             selected_expected_reward = self.env.get_expected_reward(choice)
    #             selected_expected_rewards[t] = selected_expected_reward
    #             optimal_expected_reward = self.env.get_optimal_expected_reward()
    #             optimal_expected_rewards[t] = optimal_expected_reward
    #             selected_arm_feature = self.env.arms[choice].arm_feature
    #             round_reward = self.env.play(choice)
    #             policy.update(selected_arm_feature, round_reward)
    #         expected_regrets = optimal_expected_rewards - selected_expected_rewards
    #         cum_regret[n_experiment, :] = np.cumsum(expected_regrets)[t_saved]
    #     avg_regret = np.mean(cum_regret, 0)
    #     thread_avg_regret_dict[thread_id] = avg_regret

    def multiprocessing_run_each_thread(self, thread_id, thread_num_mc, time_horizon, thread_avg_regret_dict, thread_expected_rewards_dict):
        optimal_expected_rewards = np.zeros(time_horizon)
        selected_expected_rewards = np.zeros(time_horizon)
        cum_regret = np.zeros((thread_num_mc, time_horizon))
        cum_regret_dict = {}
        avg_regret_dict = {}
        for policy in self.policies:
            cum_regret_dict[policy.name] = np.zeros((thread_num_mc, time_horizon))
            avg_regret_dict[policy.name] = np.zeros(time_horizon)
        for n_experiment in range(thread_num_mc):
            # print('thread_id, n_experiment:', thread_id, ',', n_experiment)
            self.env.init()
            # policy = self.policies[1]
            for policy in self.policies:
                self.env.re_init()
                policy.init()
                for t in range(1, time_horizon + 1):
                    self.env.update()
                    if np.isclose(self.env.expected_rewards[0], self.env.expected_rewards[1]):
                        if policy.name == 'OR-LinUCB':
                            policy.init()
                            # print('t:', t)
                        # print('expected_rewards', self.env.expected_rewards)
                    choice = policy.select_arm(self.env.arms)
                    selected_expected_reward = self.env.get_expected_reward(choice)
                    selected_expected_rewards[t - 1] = selected_expected_reward
                    optimal_expected_reward = self.env.get_optimal_expected_reward()
                    optimal_expected_rewards[t - 1] = optimal_expected_reward
                    selected_arm_feature = self.env.arms[choice].arm_feature
                    round_reward = self.env.play(choice)
                    policy.update(selected_arm_feature, round_reward)
                expected_regrets = optimal_expected_rewards - selected_expected_rewards
                cum_regret[n_experiment, :] = np.cumsum(expected_regrets)
                cum_regret_dict[policy.name][n_experiment, :] = np.cumsum(expected_regrets)
        # avg_regret = np.mean(cum_regret, 0)
        for policy in self.policies:
            avg_regret_dict[policy.name] = np.mean(cum_regret_dict[policy.name], 0)
        # print('avg_regret_dict:', avg_regret_dict)
        # thread_avg_regret_dict[thread_id] = avg_regret
        if 1 not in thread_expected_rewards_dict.keys():
            thread_expected_rewards_dict[1] = self.env.expected_rewards1
        if 2 not in thread_expected_rewards_dict.keys():
            thread_expected_rewards_dict[2] = self.env.expected_rewards2
        # print('thread_expected_rewards_dict:', thread_expected_rewards_dict)
        thread_avg_regret_dict[thread_id] = avg_regret_dict
        # print('thread_avg_regret_dict:', thread_avg_regret_dict)
        # expected_rewards_dict.append(self.env.expected_rewards2)

    def multiprocessing_run(self, num_thread, num_mc, time_horizon):
        manager = mp.Manager()
        thread_avg_regret_dict = manager.dict()
        thread_expected_rewards_dict = manager.dict()
        thread_border = []
        thread_step = num_mc // num_thread
        for i in range(num_thread):
            thread_border.append(i * thread_step)
        thread_border.append(num_mc)
        threads = []
        for i in range(num_thread):
            thread_id = i
            thread_num_mc = thread_border[i+1] - thread_border[i]
            threads.append(mp.Process(target=self.multiprocessing_run_each_thread,
                                      args=(thread_id, thread_num_mc, time_horizon, thread_avg_regret_dict, thread_expected_rewards_dict)))
            threads[i].start()
        for i in range(num_thread):
            threads[i].join()

        # avg_regret_dict = {}
        # for policy in self.policies:
        #     avg_regret_dict[policy.name] = np.zeros(time_horizon)
        # for i in range(num_thread):
        #     for policy in self.policies:
        #         avg_regret_dict[policy.name] += thread_avg_regret_dict[policy.name]
        # for policy in self.policies:
        #     avg_regret_dict[policy.name] /= num_thread

        avg_regret_dict = {}
        for policy in self.policies:
            avg_regret_dict[policy.name] = np.zeros(time_horizon)
        for policy in self.policies:
            for i in range(num_thread):
                avg_regret_dict[policy.name] += thread_avg_regret_dict[i][policy.name]
            avg_regret_dict[policy.name] /= num_thread

        expected_rewards_dict = thread_expected_rewards_dict
        return avg_regret_dict, expected_rewards_dict

        # avg_regret = np.zeros(time_horizon)
        # for i in range(num_thread):
        #     avg_regret += thread_avg_regret_dict[i]
        # avg_regret /= num_thread
        # return avg_regret

    # def run(self, n_mc, time_horizon, t_saved):
    #     optimal_expected_rewards = np.zeros(time_horizon)
    #     selected_expected_rewards = np.zeros(time_horizon)
    #     # if t_saved is None:
    #     #     t_saved = [i for i in range(time_horizon)]
    #     n_sub = len(t_saved)
    #     cum_regret = np.zeros((n_mc, n_sub))
    #     # avg_regret = np.zeros(n_sub)
    #     for n_experiment in tqdm(range(n_mc)):
    #         self.env.re_init()
    #         # print('self.env.theta:', self.env.theta)
    #         policy = self.policies[0]
    #         policy.re_init()
    #         # print('policy:', policy)
    #         # print('expected_rewards:', self.env.expected_rewards)
    #         for t in range(time_horizon):
    #             self.env.update()
    #             # print('t:', t)
    #             choice = policy.select_arm(self.env.arms)
    #             selected_expected_reward = self.env.get_expected_reward(choice)
    #             selected_expected_rewards[t] = selected_expected_reward
    #             optimal_expected_reward = self.env.get_optimal_expected_reward()
    #             optimal_expected_rewards[t] = optimal_expected_reward
    #             selected_arm_feature = self.env.arms[choice].arm_feature
    #             round_reward = self.env.play(choice)
    #             policy.update(selected_arm_feature, round_reward)
    #             # print('choice:', choice)
    #             # print('selected_expected_reward:', selected_expected_reward)
    #             # print('selected_expected_rewards:', selected_expected_rewards)
    #             # print('optimal_expected_reward:', optimal_expected_reward)
    #             # print('optimal_expected_rewards:', optimal_expected_rewards)
    #             # print('round_reward:', round_reward)
    #         expected_regrets = optimal_expected_rewards - selected_expected_rewards
    #         cum_regret[n_experiment, :] = np.cumsum(expected_regrets)[t_saved]
    #         # print('optimal_expected_rewards:', optimal_expected_rewards)
    #         # print('selected_expected_rewards:', selected_expected_rewards)
    #         # print('expected_regrets:', expected_regrets)
    #         # print('cum_regret:', cum_regret)
    #     avg_regret = np.mean(cum_regret, 0)
    #     expected_rewards1 = self.env.expected_rewards1
    #     expected_rewards2 = self.env.expected_rewards2
    #     # print('avg_regret:', avg_regret)
    #     return avg_regret, expected_rewards1, expected_rewards2