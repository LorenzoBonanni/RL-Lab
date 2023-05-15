import collections
import random
import warnings;

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

warnings.filterwarnings("ignore")
import os;

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
import gymnasium
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

def createDNN(nInputs, nOutputs, nLayer, nNodes, last_activation):
    # Initialize the neural network
    model = Sequential()
    model.add(Dense(nNodes, input_dim=nInputs, activation="relu"))  # input layer + hidden layer #1
    for _ in range(nLayer - 1):
        model.add(Dense(nNodes, activation="relu"))  # hidden layer #n
    model.add(Dense(nOutputs, activation=last_activation))  # output layer
    return model


def training_loop(env, actor_net, critic_net, updateRule, frequency=10, episodes=100):
    actor_optimizer = tf.keras.optimizers.Adam()
    critic_optimizer = tf.keras.optimizers.Adam()
    rewards_list, reward_queue = [], collections.deque(maxlen=100)
    memory_buffer = []
    length_list, length_queue = [], collections.deque(maxlen=100)
    for ep in range(episodes):
        # reset the environment and obtain the initial state
        state = env.reset()[0]
        ep_reward = 0
        ep_length = 0
        while True:
            # select the action to perform
            p = actor_net(state.reshape(-1, len(state))).numpy()[0]
            action = np.random.choice(len(p), p=p)

            # Perform the action, store the data in the memory buffer and update the reward
            next_state, reward, terminated, truncated, info = env.step(action)
            memory_buffer.append([state, action, reward, next_state, terminated])
            ep_length += 1
            ep_reward += reward

            # exit condition for the episode
            if terminated or truncated:
                length_queue.append(ep_length)
                reward_queue.append(ep_reward)
                break

            # update the current state
            state = next_state

        # if ep == 500:
        #     env = gymnasium.make('Acrobot-v1', max_episode_steps=500, render_mode='human')
        #     env = OverrideReward(env)
        # Perform the actual training
        if (ep + 1) % frequency == 0:
            updateRule(actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer)
            memory_buffer = []

        # Update the reward list to return
        length_list.append(np.mean(length_queue))
        rewards_list.append(np.mean(reward_queue))
        print(
            f"episode {ep:4d}: rw: {int(ep_reward):3d}, (averaged rw: {np.mean(reward_queue):5.2f}), len:{ep_length}, (averaged len: {np.mean(length_queue):5.2f})")

    # Close the enviornment and return the rewards list
    env.close()
    return rewards_list, length_list


def A2C(actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer, gamma=0.99):
    """
    Main update rule for the A2C update. This function includes the updates for the actor network (or policy function)
    and for the critic network (or value function)

    """
    # implement the update rule for the actor (policy function)
    # extract the information from the buffer for the policy update
    # Tape for the actor
    with tf.GradientTape() as actor_tape:
        memory_buffer = np.array(memory_buffer)
        states = np.array(list(memory_buffer[:, 0]), dtype=np.float)
        rewards = np.array(list(memory_buffer[:, 2]), dtype=np.float)
        actions = np.array(list(memory_buffer[:, 1]), dtype=int)
        next_states = np.array(list(memory_buffer[:, 3]), dtype=np.float)
        # compute the log-prob of the current trajectory and the objective function
        adv_a = rewards + gamma * critic_net(next_states).numpy().reshape(-1)
        adv_b = critic_net(states).numpy().reshape(-1)
        probs = actor_net(states)
        indices = tf.transpose(tf.stack([tf.range(probs.shape[0]), actions]))
        probs = tf.gather_nd(
            indices=indices,
            params=probs
        )
        objective = tf.math.log(probs) * (adv_a - adv_b)
        objective = - tf.math.reduce_sum(objective)
        grads = actor_tape.gradient(objective, actor_net.trainable_variables)
        actor_optimizer.apply_gradients(zip(grads, actor_net.trainable_variables))

    # update rule for the critic (value function)
    for _ in range(10):
        # Sample batch
        np.random.shuffle(memory_buffer)
        states = np.vstack(memory_buffer[:, 0])
        # actions = np.array(memory_buffer[:, 1], dtype=int)
        next_states = np.vstack(memory_buffer[:, 3])
        rewards = memory_buffer[:, 2]
        done = memory_buffer[:, 4]
        # Tape for the critic
        with tf.GradientTape() as critic_tape:
            # Compute the target and the MSE between the current prediction
            target = rewards + (1 - done.astype(int)) * gamma * critic_net(next_states)
            prediction = critic_net(states)
            objective = tf.math.square(prediction - target)
            grads = critic_tape.gradient(objective, critic_net.trainable_variables)
            critic_optimizer.apply_gradients(zip(grads, critic_net.trainable_variables))


class OverrideReward(gymnasium.wrappers.NormalizeReward):
    """
    Gymansium wrapper useful to update the reward function of the environment

    """

    def step(self, action):
        previous_observation = np.array(self.env.state, dtype=np.float32)
        observation, reward, terminated, truncated, info = self.env.step(action)
        prev_h = -np.cos(previous_observation[0]) - np.cos(previous_observation[1] + previous_observation[0])
        h = -np.cos(observation[0]) - np.cos(observation[1] + observation[0])
        if terminated:
            reward = 100
        elif h > prev_h:
            reward = h-prev_h
        else:
            reward = -1
        return observation, reward, terminated, truncated, info


def main():
    print("\n***************************************************")
    print("*  Welcome to the eleventh lesson of the RL-Lab!  *")
    print("*                 (DRL in Practice)               *")
    print("***************************************************\n")

    _training_steps = 2500

    # Crete the environment and add the wrapper for the custom reward function
    env = gymnasium.make('Acrobot-v1', max_episode_steps=500)
    env = OverrideReward(env)
    # set_seed(SEED)

    # Create the networks and perform the actual training
    actor_net = createDNN(6, 3, nLayer=2, nNodes=32, last_activation="softmax")
    critic_net = createDNN(6, 1, nLayer=2, nNodes=32, last_activation="linear")
    rewards_training, ep_lengths = training_loop(env, actor_net, critic_net, A2C, frequency=10,
                                                 episodes=_training_steps)

    # Save the trained neural network
    actor_net.save("AcroBotActor.h5")

    # Plot the results
    t = np.arange(0, _training_steps)
    plt.plot(t, ep_lengths, label="A2C", linewidth=3)
    plt.xlabel("epsiodes", fontsize=16)
    plt.ylabel("length", fontsize=16)
    plt.legend()
    plt.title("RWRD2")
    plt.show()

    plt.plot(t, rewards_training, label="A2C", linewidth=3)
    plt.xlabel("epsiodes", fontsize=16)
    plt.ylabel("rewards", fontsize=16)
    plt.legend()
    plt.title("RWRD2")
    plt.show()

if __name__ == "__main__":
    main()
