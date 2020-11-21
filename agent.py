import numpy as np
from replay_memory import ReplayBuffer
from network import ActionValueNetwork
from adam import Adam
from copy import deepcopy



class Agent():
    def __init__(self):
        self.name = "expected_sarsa_agent"

    def agent_init(self, agent_config):
        self.replay_buffer = ReplayBuffer(agent_config['replay_buffer_size'],
                                          agent_config['minibatch_sz'], agent_config.get("seed"))
        self.network = ActionValueNetwork(agent_config['network_config'])
        self.optimizer = Adam(self.network.layer_sizes,
                              agent_config["optimizer_config"])
        self.num_actions = agent_config['network_config']['num_actions']
        self.num_replay = agent_config['num_replay_updates_per_step']
        self.discount = agent_config['gamma']
        self.tau = agent_config['tau']

        self.rand_generator = np.random.RandomState(agent_config.get("seed"))

        self.last_state = None
        self.last_action = None

        self.sum_rewards = 0
        self.episode_steps = 0

    def policy(self, state):
        action_values = self.network.get_action_values(state)
        probs_batch = softmax(action_values, self.tau)
        action = self.rand_generator.choice(
            self.num_actions, p=probs_batch.squeeze())
        return action

    def agent_start(self, state):
        self.sum_rewards = 0
        self.episode_steps = 0
        self.last_state = np.array([state])
        self.last_action = self.policy(self.last_state)
        return self.last_action

    def agent_step(self, reward, state):
        self.sum_rewards += reward
        self.episode_steps += 1

        state = np.array([state])

        action = self.policy(state)

        self.replay_buffer.append(
            self.last_state, self.last_action, reward, 0, state)

        # Perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            current_q = deepcopy(self.network)
            for _ in range(self.num_replay):

                # Get sample experiences from the replay buffer
                experiences = self.replay_buffer.sample()

                # Call optimize_network to update the weights of the network
                optimize_network(experiences, self.discount,
                                 self.optimizer, self.network, current_q, self.tau)

        # Update the last state and last action.
        self.last_state = state
        self.last_action = action

        return action

    # update of the weights using optimize_network
    def agent_end(self, reward):
        self.sum_rewards += reward
        self.episode_steps += 1

        # Set terminal state to an array of zeros
        state = np.zeros_like(self.last_state)

        # Append new experience to replay buffer
        self.replay_buffer.append(
            self.last_state, self.last_action, reward, 1, state)

        # Perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            current_q = deepcopy(self.network)
            for _ in range(self.num_replay):

                # Get sample experiences from the replay buffer
                experiences = self.replay_buffer.sample()

                # Call optimize_network to update the weights of the network
                optimize_network(experiences, self.discount,
                                 self.optimizer, self.network, current_q, self.tau)

    def agent_message(self, message):
        if message == "get_sum_reward":
            return self.sum_rewards
        else:
            raise Exception("Unrecognized Message!")


def optimize_network(experiences, discount, optimizer, network, current_q, tau):
    # Get states, action, rewards, terminals, and next_states from experiences
    states, actions, rewards, terminals, next_states = map(
        list, zip(*experiences))
    states = np.concatenate(states)
    next_states = np.concatenate(next_states)
    rewards = np.array(rewards)
    terminals = np.array(terminals)
    batch_size = states.shape[0]

    # Compute TD error using the get_td_error function
    delta_vec = get_td_error(states, next_states, actions,
                             rewards, discount, terminals, network, current_q, tau)

    # Batch Indices is an array from 0 to the batch_size - 1.
    batch_indices = np.arange(batch_size)

    # Make a td error matrix of shape (batch_size, num_actions)
    # delta_mat has non-zero value only for actions taken
    delta_mat = np.zeros((batch_size, network.num_actions))
    delta_mat[batch_indices, actions] = delta_vec

    # Pass delta_mat to compute the TD errors times the gradients of the network's weights from back-propagation
    td_update = network.get_TD_update(states, delta_mat)

    # Pass network.get_weights and the td_update to the optimizer to get updated weights
    weights = optimizer.update_weights(network.get_weights(), td_update)

    network.set_weights(weights)


def softmax(action_values, tau=1.0):

    preferences = action_values/tau

    max_preference = np.max(preferences, axis=1)

    reshaped_max_preference = max_preference.reshape((-1, 1))

    exp_preferences = np.exp(preferences - reshaped_max_preference)
    #print(f"exp Pref {exp_preferences.shape}")

    sum_of_exp_preferences = np.sum(exp_preferences, axis=1)
    #print(f"sum Pref {sum_of_exp_preferences.shape}")

    reshaped_sum_of_exp_preferences = sum_of_exp_preferences.reshape((-1, 1))

    # Compute the action probabilities according to the equation in the previous cell.
    action_probs = exp_preferences / reshaped_sum_of_exp_preferences

    action_probs = action_probs.squeeze()
    return action_probs


def get_td_error(states, next_states, actions, rewards, discount, terminals, network, current_q, tau):

    q_next_mat = current_q.get_action_values(next_states)
    #print(q_next_mat.shape)
    #print(states.shape[0])

    # Compute policy at next state by passing the action-values in q_next_mat to softmax()
    probs_mat = softmax(q_next_mat, tau)
    #print(probs_mat.shape)

    # Compute the estimate of the next state value, v_next_vec.
    v_next_vec = (q_next_mat*probs_mat).sum(axis=1) * (1-terminals)
    #print(v_next_vec)

    # Compute Expected Sarsa target
    target_vec = rewards + (discount*v_next_vec)

    # Compute action values at the current states for all actions using network

    q_mat = network.get_action_values(states)

    # Batch Indices is an array from 0 to the batch size - 1.
    batch_indices = np.arange(q_mat.shape[0])

    # Compute q_vec by selecting q(s, a) from q_mat for taken actions
    q_vec = []
    for batch in batch_indices:
        #print(batch)
        q_vec.append(q_mat[batch][actions[batch]])
    # Compute TD errors for actions taken
    delta_vec = target_vec - q_vec

    return delta_vec
