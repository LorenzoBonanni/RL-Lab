from typing import List, Tuple

import warnings;

warnings.filterwarnings("ignore");
import os;

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3';

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import gymnasium, collections
from gymnasium.core import ObsType, ActType, SupportsFloat


class BufferHelper:

    def __init__(self) -> None:
        self.memory_buffer = []
        self.current_trajectory = []

    def store_step(self, state: ObsType, action: ActType, reward: SupportsFloat) -> None:
        self.current_trajectory.append([state, action, reward])

    def get_length(self) -> int:
        return len(self.memory_buffer)

    def end_trajectory(self):
        self.memory_buffer.append(self.current_trajectory)
        self.current_trajectory = []

    def clear(self) -> None:
        self.memory_buffer = []

    def get_splitted_infos(self, idx: int) -> Tuple[List, List, List]:

        trajectory = self.memory_buffer[idx]
        if trajectory is None: return [], [], []

        states = np.array([step[0] for step in trajectory])
        actions = np.array([step[1] for step in trajectory])
        rewards = np.array([step[2] for step in trajectory])

        return states, actions, rewards

    def compute_discount_norm(rewards: np.ndarray, gamma: float) -> np.ndarray:

        # Compute discount
        for i in range(len(rewards) - 2, 0, -1):
            rewards[i] += rewards[i + 1] * gamma

        # Normalization with mean and standard deviation
        rewards -= np.mean(rewards)
        rewards /= np.std(rewards + 1e-7)

        rewards.reshape(-1, 1)

        return rewards


def createDNN(nInputs, nOutputs, nLayer, nNodes):
    # Initialize the neural network
    model = Sequential()
    model.add(Dense(nNodes, input_dim=nInputs, activation="relu"))  # input layer + hidden layer #1
    for _ in range(nLayer - 1):
        model.add(Dense(nNodes, activation="relu"))  # hidden layer #n
    model.add(Dense(nOutputs, activation="softmax"))  # output layer
    return model


def training_loop(env, neural_net, updateRule, frequency=10, episodes=100):
    """
    Main loop of the reinforcement learning algorithm. Execute the actions and interact
    with the environment to collect the experience for the training.
    Args:
        env: gymnasium environment for the training
        neural_net: the model to train
        updateRule: external function for the training of the neural network
    Returns:
        averaged_rewards: array with the averaged rewards obtained
    """

    # initialize the optimizer
    optimizer = tf.keras.optimizers.Adam()
    rewards_list, reward_queue = [], collections.deque(maxlen=100)
    memory_buffer = BufferHelper()

    for ep in range(episodes):
        # reset the environment and obtain the initial state
        state = env.reset()[0]
        ep_reward = 0
        while True:

            # select the action to perform
            p = neural_net(state.reshape(-1, 4)).numpy()[0]
            action = np.random.choice(2, p=p)

            # Perform the action, store the data in the memory buffer and update the reward
            next_state, reward, terminated, truncated, _ = env.step(action)
            memory_buffer.store_step(state, action, reward)
            ep_reward += reward

            # exit condition for the episode
            if terminated or truncated:
                break

            # update the current state
            state = next_state

        memory_buffer.end_trajectory()
        # Perform the actual training every 'frequency' episodes
        if (ep + 1) % frequency == 0:
            updateRule(neural_net, memory_buffer, optimizer)
            memory_buffer.clear()

        # Update the reward list to return
        reward_queue.append(ep_reward)
        rewards_list.append(np.mean(reward_queue))
        print(f"episode {ep:4d}: rw: {int(ep_reward):3d} (averaged: {np.mean(reward_queue):5.2f})")

    # Close the enviornment and return the rewards list
    env.close()
    return rewards_list


def REINFORCE_naive(neural_net, memory_buffer: BufferHelper, optimizer, gamma=0.999):
    """
    Main update rule for the REINFORCE process, the naive implementation of the policy-gradient theorem.
    """
    with tf.GradientTape() as tape:
        objectives = []
        for i in range(memory_buffer.get_length()):
            states, actions, rewards = memory_buffer.get_splitted_infos(i)
            probs = neural_net(states)

            idxs = np.array([[i, action] for i, action in enumerate(actions)])
            action_probs = tf.expand_dims(tf.gather_nd(probs, idxs), axis=-1)

            log_probs = tf.reduce_sum(tf.math.log(action_probs))

            objectives.append(tf.multiply(tf.constant(sum(rewards), dtype=tf.float32), log_probs))

        objective = -tf.math.reduce_mean(objectives)
    grad = tape.gradient(objective, neural_net.trainable_variables)
    optimizer.apply_gradients(zip(grad, neural_net.trainable_variables))


def REINFORCE_rw2go(neural_net, memory_buffer, optimizer):
    """
    Main update rule for the REINFORCE process, with the addition of the reward-to-go trick,
    """

    with tf.GradientTape() as tape:
        objectives = []
        for i in range(memory_buffer.get_length()):
            states, actions, rewards = memory_buffer.get_splitted_infos(i)

            probs = neural_net(states)
            idxs = np.array([[i, action] for i, action in enumerate(actions)])
            action_probs = tf.expand_dims(tf.gather_nd(probs, idxs), axis=-1)

            log_probs = tf.math.log(action_probs)
            cum_sum = tf.reshape(tf.cumsum(rewards), (1, -1))

            reverse = tf.cast(tf.reverse(cum_sum, axis=[-1]), tf.float32)
            log_probs = tf.reshape(log_probs, (1, -1))

            objectives.append(tf.reduce_sum(tf.multiply(log_probs, reverse)))

        objective = -tf.math.reduce_mean(objectives)
        grad = tape.gradient(objective, neural_net.trainable_variables)
        optimizer.apply_gradients(zip(grad, neural_net.trainable_variables))


def main():
    print("\n*************************************************")
    print("*  Welcome to the ninth lesson of the RL-Lab!   *")
    print("*                 (REINFORCE)                   *")
    print("*************************************************\n")

    _training_steps = 5000
    env = gymnasium.make("CartPole-v1")  # render_mode="human" )

    # Training A)
    # neural_net = createDNN(4, 2, nLayer=2, nNodes=32)
    # rewards_naive = training_loop(env, neural_net, REINFORCE_naive, episodes=_training_steps)

    # Training B)
    neural_net = createDNN(4, 2, nLayer=2, nNodes=32)
    rewards_rw2go = training_loop(env, neural_net, REINFORCE_rw2go, episodes=_training_steps)

    # Plot
    t = np.arange(0, _training_steps)
    # plt.plot(t, rewards_naive, label="naive", linewidth=3)
    plt.plot(t, rewards_rw2go, label="reward to go", linewidth=3)
    plt.xlabel("epsiodes", fontsize=16)
    plt.ylabel("reward", fontsize=16)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()