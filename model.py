import gym
from collections import deque
from copy import deepcopy
from tqdm import tqdm
import os
import gym
import shutil
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent

class EnvManager:

    def __init__(self, env_class, agent_class):
        self.environment = env_class()
        self.agent = agent_class()

        self.total_reward = None
        self.last_action = None
        self.num_steps = None
        self.num_episodes = None

    def init(self, agent_init_info={}, env_init_info={}):
        self.agent.agent_init(agent_init_info)

        self.total_reward = 0.0
        self.num_steps = 0
        self.num_episodes = 0

    def start(self, agent_start_info={}, env_start_info={}):

        self.total_reward = 0.0
        self.num_steps = 1

        last_state = self.environment.env_start()
        self.last_action = self.agent.agent_start(last_state)

        observation = (last_state, self.last_action)

        return observation

    def agent_start(self, observation):
        return self.agent.agent_start(observation)

    def agent_step(self, reward, observation):
        return self.agent.agent_step(reward, observation)

    def agent_end(self, reward):
        self.agent.agent_end(reward)

    def env_start(self):
        self.total_reward = 0.0
        self.num_steps = 1

        this_observation = self.environment.env_start()

        return this_observation

    def env_step(self, action):
        ro = self.environment.env_step(action)
        (this_reward, _, terminal) = ro

        self.total_reward += this_reward

        if terminal:
            self.num_episodes += 1
        else:
            self.num_steps += 1

        return ro

    def step(self):
        (reward, last_state, term) = self.environment.env_step(self.last_action)

        self.total_reward += reward

        if term:
            self.num_episodes += 1
            self.agent.agent_end(reward)
            roat = (reward, last_state, None, term)
        else:
            self.num_steps += 1
            self.last_action = self.agent.agent_step(reward, last_state)
            roat = (reward, last_state, self.last_action, term)

        return roat

    def cleanup(self):
        self.environment.env_cleanup()
        self.agent.agent_cleanup()

    def agent_message(self, message):
        return self.agent.agent_message(message)

    def env_message(self, message):
        return self.environment.env_message(message)

    def episode(self, max_steps_this_episode):
        is_terminal = False

        self.start()

        while (not is_terminal) and ((max_steps_this_episode == 0) or
                                     (self.num_steps < max_steps_this_episode)):
            rl_step_result = self.step()
            is_terminal = rl_step_result[3]

        return is_terminal

    def return_reward(self):
        return self.total_reward

    def num_steps(self):
        return self.num_steps

    def num_episodes(self):
        return self.num_episodes


class LunarLanderEnvironment():
    def __init__(self, env_info={}):
        self.reward = None
        self.observation = None
        self.termination = None
        self.reward_obs_term = None
        self.env = gym.make("LunarLander-v2")
        self.env.seed(0)

    def env_start(self):
        self.reward = 0.0
        self.observation = self.env.reset()
        self.is_terminal = False

        self.reward_obs_term = (
            self.reward, self.observation, self.is_terminal)

        # return first state observation from the environment
        return self.reward_obs_term[1]

    def env_step(self, action):
        last_state = self.reward_obs_term[1]
        current_state, reward, is_terminal, _ = self.env.step(action)
        self.env.render()
        self.reward_obs_term = (reward, current_state, is_terminal)

        return self.reward_obs_term


plt_legend_dict = {"expected_sarsa_agent": "Expected SARSA with neural network",
                   "random_agent": "Random"}

plt_label_dict = {"expected_sarsa_agent": "Sum of\nreward\nduring\nepisode"}


def smooth(data, k):
    num_episodes = data.shape[1]
    num_runs = data.shape[0]

    smoothed_data = np.zeros((num_runs, num_episodes))

    for i in range(num_episodes):
        if i < k:
            smoothed_data[:, i] = np.mean(data[:, :i+1], axis=1)
        else:
            smoothed_data[:, i] = np.mean(data[:, i-k:i+1], axis=1)

    return smoothed_data

experiment_parameters = {
    "num_episodes": 500,
    "timeout": 1000,
    "num_runs": 1
}

# Environment parameters
environment_parameters = {}

environment = LunarLanderEnvironment

# Agent parameters
agent_parameters = {
    'network_config': {
        'state_dim': 8,
        'num_hidden_units': 256,
        'num_actions': 4
    },
    'optimizer_config': {
        'step_size': 1e-3,
        'beta_m': 0.9,
        'beta_v': 0.999,
        'epsilon': 1e-8
    },
    'replay_buffer_size': 50000,
    'minibatch_sz': 16,
    'num_replay_updates_per_step': 8,
    'gamma': 0.99,
    'tau': 0.001
}
agent = Agent

env = EnvManager(environment, agent)
    
# save sum of reward at the end of each episode
agent_sum_reward = np.zeros((experiment_parameters["num_runs"],
                             experiment_parameters["num_episodes"]))
env_info = {}

agent_info = agent_parameters

for run in range(1, experiment_parameters["num_runs"]+1):
    agent_info["seed"] = run
    agent_info["network_config"]["seed"] = run
    env_info["seed"] = run

    env.init(agent_info, env_info)
    
    for episode in tqdm(range(1, experiment_parameters["num_episodes"]+1)):
        # run episode
        env.episode(experiment_parameters["timeout"])
        
        episode_reward = env.agent_message("get_sum_reward")
        agent_sum_reward[run - 1, episode - 1] = episode_reward

save_name = "{}".format(env.agent.name)
if not os.path.exists('results'):
    os.makedirs('results')
np.save("results/sum_reward_{}".format(save_name), agent_sum_reward)

data_name_array = ["expected_sarsa_agent"]

path_dict = {"expected_sarsa_agent": "results/",
             "random_agent": "./"}

#plot the results
plt_agent_sweeps = []

fig, ax = plt.subplots(figsize=(8, 6))

for data_name in data_name_array:

    # load data
    filename = 'sum_reward_{}'.format(data_name).replace('.', '')
    sum_reward_data = np.load(
        '{}/{}.npy'.format(path_dict[data_name], filename))

    # smooth data
    smoothed_sum_reward = smooth(data=sum_reward_data, k=100)

    mean_smoothed_sum_reward = np.mean(smoothed_sum_reward, axis=0)

    plot_x_range = np.arange(0, mean_smoothed_sum_reward.shape[0])
    graph_current_agent_sum_reward, = ax.plot(
        plot_x_range, mean_smoothed_sum_reward[:], label=plt_legend_dict[data_name])
    plt_agent_sweeps.append(graph_current_agent_sum_reward)

ax.legend(handles=plt_agent_sweeps, fontsize=13)
ax.set_title("Learning Curve", fontsize=15)
ax.set_xlabel('Episodes', fontsize=14)
ax.set_ylabel(plt_label_dict[data_name_array[0]],
                rotation=0, labelpad=40, fontsize=14)
ax.set_ylim([-300, 300])

plt.tight_layout()
plt.show()

env.environment.env.close()
