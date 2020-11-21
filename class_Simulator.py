import numpy as np
from tqdm import tqdm
import multiprocessing as mp


class Simulator(object):
    def __init__(self, env, policies):
        self.env = env
        self.policies = policies

    def run(self, n_mc, time_horizon, t_saved):
        optimal_expected_rewards = np.zeros(time_horizon)
        selected_expected_rewards = np.zeros(time_horizon)
        # if t_saved is None:
        #     t_saved = [i for i in range(time_horizon)]
        n_sub = len(t_saved)
        cum_regret = np.zeros((n_mc, n_sub))
        # avg_regret = np.zeros(n_sub)
        for n_experiment in tqdm(range(n_mc)):
            self.env.re_init()
            # print('self.env.theta:', self.env.theta)
            policy = self.policies[0]
            policy.re_init()
            # print('policy:', policy)
            # print('expected_rewards:', self.env.expected_rewards)
            for t in range(time_horizon):
                self.env.update()
                # print('t:', t)
                choice = policy.select_arm(self.env.arms)
                selected_expected_reward = self.env.get_expected_reward(choice)
                selected_expected_rewards[t] = selected_expected_reward
                optimal_expected_reward = self.env.get_optimal_expected_reward()
                optimal_expected_rewards[t] = optimal_expected_reward
                selected_arm_feature = self.env.arms[choice].arm_feature
                round_reward = self.env.play(choice)
                policy.update(selected_arm_feature, round_reward)
                # print('choice:', choice)
                # print('selected_expected_reward:', selected_expected_reward)
                # print('selected_expected_rewards:', selected_expected_rewards)
                # print('optimal_expected_reward:', optimal_expected_reward)
                # print('optimal_expected_rewards:', optimal_expected_rewards)
                # print('round_reward:', round_reward)
            expected_regrets = optimal_expected_rewards - selected_expected_rewards
            cum_regret[n_experiment, :] = np.cumsum(expected_regrets)[t_saved]
            # print('optimal_expected_rewards:', optimal_expected_rewards)
            # print('selected_expected_rewards:', selected_expected_rewards)
            # print('expected_regrets:', expected_regrets)
            # print('cum_regret:', cum_regret)
        avg_regret = np.mean(cum_regret, 0)
        # print('avg_regret:', avg_regret)
        return avg_regret

    def multiprocessing_run_each_thread(self, thread_id, thread_num_mc, time_horizon, t_saved, thread_avg_regret_dict):
        optimal_expected_rewards = np.zeros(time_horizon)
        selected_expected_rewards = np.zeros(time_horizon)
        n_sub = len(t_saved)
        cum_regret = np.zeros((thread_num_mc, n_sub))
        for n_experiment in range(thread_num_mc):
            self.env.re_init()
            policy = self.policies[0]
            policy.re_init()
            for t in range(time_horizon):
                self.env.update()
                choice = policy.select_arm(self.env.arms)
                selected_expected_reward = self.env.get_expected_reward(choice)
                selected_expected_rewards[t] = selected_expected_reward
                optimal_expected_reward = self.env.get_optimal_expected_reward()
                optimal_expected_rewards[t] = optimal_expected_reward
                selected_arm_feature = self.env.arms[choice].arm_feature
                round_reward = self.env.play(choice)
                policy.update(selected_arm_feature, round_reward)
            expected_regrets = optimal_expected_rewards - selected_expected_rewards
            cum_regret[n_experiment, :] = np.cumsum(expected_regrets)[t_saved]
        avg_regret = np.mean(cum_regret, 0)
        thread_avg_regret_dict[thread_id] = avg_regret

    def multiprocessing_run(self, num_thread, num_mc, time_horizon, t_saved):
        manager = mp.Manager()
        thread_avg_regret_dict = manager.dict()
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
                                      args=(thread_id, thread_num_mc, time_horizon, t_saved, thread_avg_regret_dict)))
            threads[i].start()
        for i in range(num_thread):
            threads[i].join()
        avg_regret = np.zeros(len(t_saved))
        for i in range(num_thread):
            avg_regret += thread_avg_regret_dict[i]
        avg_regret /= num_thread
        return avg_regret

