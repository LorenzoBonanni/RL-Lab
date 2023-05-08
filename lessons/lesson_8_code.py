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

# Setting the seeds
SEED = 15


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    # When running on the CuDNN backend, two further options must be set
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def epsilon_greedy(q, state, epsilon):
    """
    Epsilon-greedy action selection function

    Args:
        q: q table
        state: agent's current state
        epsilon: epsilon parameter

    Returns:
        action id
    """
    q = q.numpy()
    if np.random.random() < epsilon:
        return np.random.choice(q.shape[1])
    # return q[state].argmax()
    return q.argmax()


def createDNN(nInputs, nOutputs, nLayer, nNodes):
    """
    Function that generates a neural network with the given requirements.

    Args:
        nInputs: number of input nodes
        nOutputs: number of output nodes
        nLayer: number of hidden layers
        nNodes: number nodes in the hidden layers

    Returns:
        model: the generated tensorflow model

    """

    # Initialize the neural network
    model = Sequential()
    model.add(Dense(nNodes, input_dim=nInputs, activation="relu"))  # input layer + hidden layer #1
    for _ in range(nLayer - 1):
        model.add(Dense(nNodes, activation="relu"))  # hidden layer #n
    model.add(Dense(nOutputs, activation="linear"))  # output layer
    return model


def mse(network, dataset_input, target):
    """
    Compute the MSE loss function

    """

    # Compute the predicted value, over time this value should
    # looks more like to the expected output (i.e., target)
    predicted_value = network(dataset_input)

    # Compute MSE between the predicted value and the expected labels
    mse = tf.math.square(predicted_value - target)
    mse = tf.math.reduce_mean(mse)

    # Return the averaged values for computational optimization
    return mse


def training_loop(env, neural_net, updateRule, eps=1, episodes=100, updates=10):
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
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    rewards_list, memory_buffer = [], collections.deque(maxlen=1000)
    averaged_rewards = []
    for ep in range(episodes):
        eps *= 0.999
        # reset the environment and obtain the initial state
        state = env.reset(seed=SEED)[0]
        ep_reward = 0
        while True:

            # select the action to perform
            action = epsilon_greedy(
                q=neural_net(state.reshape(-1, 4)),
                state=state,
                epsilon=eps
            )

            # Perform the action, store the data in the memory buffer and update the reward
            next_state, reward, terminated, truncated, info = env.step(action)
            memory_buffer.append([state, action, reward, next_state, terminated])
            ep_reward += reward

            # Perform the actual training
            for _ in range(1):
                updateRule(neural_net, memory_buffer, optimizer)

            # exit condition for the episode
            if terminated or truncated:
                break

            # update the current state
            state = next_state

        eps = eps * 0.99
        # Update the reward list to return
        rewards_list.append(ep_reward)
        averaged_rewards.append(np.mean(rewards_list))
        print(f"episode {ep:2d}: rw: {averaged_rewards[-1]:3.2f}, eps: {eps:3.2f}")

    # Close the enviornment and return the rewards list
    env.close()
    return rewards_list


def DQNUpdate(neural_net, memory_buffer, optimizer, batch_size=32, gamma=0.99):
    """
    Main update rule for the DQN process. Extract data from the memory buffer and update
    the network computing the gradient.

    """

    if len(memory_buffer) < batch_size:
        return

    indices = np.random.randint(len(memory_buffer), size=batch_size)
    memory_buffer = np.array(list(memory_buffer))
    # extract data from the buffer
    batch = memory_buffer[indices, :]
    state = np.array(list(batch[:, 0]), dtype=np.float)
    action = np.array(list(batch[:, 1]), dtype=np.int)
    reward = np.array(list(batch[:, 2]), dtype=np.float)
    next_state = np.array(list(batch[:, 3]), dtype=np.float)
    done = np.array(list(batch[:, 4]), dtype=bool)

    # compute the target for the training
    target = neural_net(state).numpy()
    not_done = np.logical_not(done)
    idx_done = np.where(done)[0]
    idx_not_done = np.where(not_done)[0]
    action_done = action[idx_done]
    action_not_done = action[idx_not_done]
    target[idx_done, action_done] = reward[idx_done]

    max_q = tf.math.reduce_max(neural_net(next_state)).numpy()
    target[idx_not_done, action_not_done] = reward[idx_not_done] + (max_q * gamma)

    # compute the gradient and perform the backpropagation step
    with tf.GradientTape() as tape:
        objective = mse(neural_net, state, target)
        grad = tape.gradient(objective, neural_net.trainable_variables)
        optimizer.apply_gradients(zip(grad, neural_net.trainable_variables))


def main():
    print("\n************************************************")
    print("*  Welcome to the eighth lesson of the RL-Lab!   *")
    print("*               (Deep Q-Network)                 *")
    print("**************************************************\n")

    set_seed(SEED)

    _training_steps = 100

    env = gymnasium.make("CartPole-v1", render_mode="human" )
    neural_net = createDNN(4, 2, nLayer=2, nNodes=32)
    rewards = training_loop(env, neural_net, DQNUpdate, episodes=_training_steps)

    t = np.arange(0, _training_steps)
    plt.plot(t, rewards, label="eps: 0", linewidth=3)
    plt.xlabel("episodes", fontsize=16)
    plt.ylabel("reward", fontsize=16)
    plt.show()


if __name__ == "__main__":
    main()
