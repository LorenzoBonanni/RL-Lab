import random
import warnings

warnings.filterwarnings("ignore")
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gymnasium, collections

SEED = 15


# def set_seed(seed: int = 42) -> None:
#     random.seed(seed)
#     np.random.seed(seed)
#     tf.random.set_seed(seed)
#     tf.experimental.numpy.random.seed(seed)
#     # When running on the CuDNN backend, two further options must be set
#     os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
#     os.environ['TF_DETERMINISTIC_OPS'] = '1'
#     # Set a fixed value for the hash seed
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     print(f"Random seed set as {seed}")


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
    memory_buffer = []
    for ep in range(episodes):
        trajectory = []
        # reset the environment and obtain the initial state
        state = env.reset()[0]
        ep_reward = 0
        while True:

            # select the action to perform
            p = neural_net(state.reshape(-1, 4)).numpy()
            action = random.choices(
                population=range(len(p)),
                weights=p
            )[0]

            # Perform the action, store the data in the memory buffer and update the reward
            next_state, reward, terminated, truncated, info = env.step(action)
            trajectory.append([state, action, reward, next_state, terminated])
            ep_reward += reward

            # exit condition for the episode
            if terminated or truncated:
                break

            # update the current state
            state = next_state

        memory_buffer.append(trajectory)
        # Perform the actual training every 'frequency' episodes
        if (ep + 1) % frequency == 0:
            updateRule(neural_net, memory_buffer, optimizer)
            memory_buffer = []

        # Update the reward list to return
        reward_queue.append(ep_reward)
        rewards_list.append(np.mean(reward_queue))
        print(f"episode {ep:4d}: rw: {int(ep_reward):3d} (averaged: {np.mean(reward_queue):5.2f})")

    # Close the enviornment and return the rewards list
    env.close()
    return rewards_list


def REINFORCE_naive(neural_net, memory_buffer, optimizer):
    """
    Main update rule for the REINFORCE process, the naive implementation of the policy-gradient theorem.

    """
    for trj in memory_buffer:
        trj = np.array(trj)
        states = np.array(list(trj[:, 0]), dtype=np.float)
        rewards = trj[:, 2]

        # Initialize the array for the objectives, one for each episode considered

        # compute the gradient and perform the backpropagation step
        with tf.GradientTape() as tape:
            probs = neural_net(states)
            log_prob_sum = tf.reduce_sum(tf.math.log(probs), axis=1)
            objectives = log_prob_sum * sum(rewards)
            # Implement the update rule, notice that the REINFORCE objective
            objective = -tf.math.reduce_mean(objectives)
            grad = tape.gradient(objective, neural_net.trainable_variables)
            optimizer.apply_gradients(zip(grad, neural_net.trainable_variables))


def REINFORCE_rw2go(neural_net, memory_buffer, optimizer):
    """
    Main update rule for the REINFORCE process, with the addition of the reward-to-go trick,

    """
    for trj in memory_buffer:
        trj = np.array(trj)
        np.random.shuffle(trj)
        states = np.array(list(trj[:, 0]), dtype=np.float)
        rewards = np.array(list(trj[:, 2]), dtype=np.float)

        # compute the gradient and perform the backpropagation step
        with tf.GradientTape() as tape:
            probs = neural_net(states)
            log_prob = tf.cast(tf.math.log(probs), tf.float64)
            rwrds = tf.reverse(tf.cumsum(tf.reverse(rewards, axis=[0])), axis=[0])
            mul = tf.multiply(log_prob, tf.expand_dims(rwrds, axis=1))
            objectives = tf.reduce_sum(mul, axis=1)
            # Implement the update rule, notice that the REINFORCE objective
            objective = -tf.math.reduce_mean(objectives)
            grad = tape.gradient(objective, neural_net.trainable_variables)
            optimizer.apply_gradients(zip(grad, neural_net.trainable_variables))


def main():
    print("\n*************************************************")
    print("*  Welcome to the ninth lesson of the RL-Lab!   *")
    print("*                 (REINFORCE)                   *")
    print("*************************************************\n")

    _training_steps = 1500
    env = gymnasium.make("CartPole-v1")

    # Training A)
    neural_net = createDNN(4, 2, nLayer=2, nNodes=32)
    rewards_naive = training_loop(env, neural_net, REINFORCE_naive, episodes=_training_steps)

    # Training B)
    neural_net = createDNN(4, 2, nLayer=2, nNodes=32)
    rewards_rw2go = training_loop(env, neural_net, REINFORCE_rw2go, episodes=_training_steps)

    # Plot
    t = np.arange(0, _training_steps)
    plt.plot(t, rewards_naive, label="naive", linewidth=3)
    plt.plot(t, rewards_rw2go, label="reward to go", linewidth=3)
    plt.xlabel("epsiodes", fontsize=16)
    plt.ylabel("reward", fontsize=16)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
