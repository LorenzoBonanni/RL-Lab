import collections
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

MAP_SIZE = 4
n_actions = 4
MAX_DIST = np.abs(MAP_SIZE - 1 - 0) + np.abs(MAP_SIZE - 1 - 0)


def next_state_distance(row, column, action):
    target_column = MAP_SIZE - 1
    target_row = MAP_SIZE - 1
    new_column = None
    new_row = None

    # 0: LEFT
    if action == 0:
        new_column = column - 1
        new_row = row
    # 1: DOWN
    elif action == 1:
        new_column = column
        new_row = row + 1
    # 2: RIGHT
    elif action == 2:
        new_column = column + 1
        new_row = row
    # 3: UP
    elif action == 3:
        new_column = column
        new_row = row - 1

    if new_column < 0 or new_column > MAP_SIZE - 1 or new_row < 0 or new_row > MAP_SIZE - 1:
        new_column = column
        new_row = row

    # return new_row*4+new_column
    return np.abs(target_column - new_column) + np.abs(target_row - new_row)


# heuristic dictionary
distances = {
    r * MAP_SIZE + c: [
        next_state_distance(r, c, a)
        for a in range(n_actions)
    ]
    for c in range(MAP_SIZE)
    for r in range(MAP_SIZE)
}
s_unpack = {
    r * MAP_SIZE + c: [r, c]
    for c in range(MAP_SIZE)
    for r in range(MAP_SIZE)
}


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
    # length_list = []
    succ_rates = []
    succ_rate = 0
    count = 0
    for ep in range(episodes):
        # reset the environment and obtain the initial state
        state = env.reset()[0]
        ep_reward = 0
        # ep_length = 0
        while True:
            # select the action to perform
            p = actor_net(np.array([[state]])).numpy()[0]
            action = np.random.choice(len(p), p=p)

            # Perform the action, store the data in the memory buffer and update the reward
            next_state, reward, terminated, truncated, info = env.step(action)
            memory_buffer.append([state, action, reward, next_state, terminated])
            # ep_length += 1
            ep_reward += reward

            # exit condition for the episode
            if terminated or truncated:
                # length_list.append(ep_length)
                succ_rate += int(next_state == MAP_SIZE-1 * MAP_SIZE + MAP_SIZE-1)
                # print(reward)
                break

            # update the current state
            state = next_state
        # Perform the actual training
        if (ep + 1) % frequency == 0:
            updateRule(actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer)
            memory_buffer = []
            succ_rates.append(succ_rate)

        # Update the reward list to return
        reward_queue.append(ep_reward)
        rewards_list.append(np.mean(reward_queue))
        print(f"episode {ep:4d}: rw: {int(ep_reward):3d} succ:{succ_rate} (averaged: {np.mean(reward_queue):5.2f})")
        if (ep + 1) % frequency == 0:
            succ_rate = 0


    # Close the environment and return the rewards list
    env.close()
    return rewards_list, succ_rates


def A2C(actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer, gamma=0.99):
    """
    Main update rule for the A2C update. This function includes the updates for the actor network (or policy function)
    and for the critic network (or value function)

    """
    # implement the update rule for the actor (policy function)
    # extract the information from the buffer for the policy update
    # Tape for the actor
    objectives = []
    with tf.GradientTape() as actor_tape:
        memory_buffer = np.array(memory_buffer)
        states = np.vstack(memory_buffer[:, 0])
        rewards = memory_buffer[:, 2]
        actions = memory_buffer[:, 1].astype(int)
        next_states = np.vstack(memory_buffer[:, 3])
        # compute the log-prob of the current trajectory and the objective function
        adv_a = rewards + gamma * critic_net(next_states).numpy().reshape(-1)
        adv_b = critic_net(states).numpy().reshape(-1)
        probs = actor_net(states)
        indices = tf.transpose(
            tf.stack([tf.range(probs.shape[0]), actions])
        )
        probs = tf.gather_nd(
            indices=indices,
            params=probs
        )
        objective = tf.math.log(probs) * (adv_a - adv_b)
        objectives.append(tf.reduce_mean(tf.reduce_sum(objective)))

        objective = tf.math.log(probs) * (adv_a - adv_b)
        objective = -tf.math.reduce_sum(objective)
        grads = actor_tape.gradient(objective, actor_net.trainable_variables)
        actor_optimizer.apply_gradients(zip(grads, actor_net.trainable_variables))

    # update rule for the critic (value function)
    for _ in range(10):
        # Sample batch
        np.random.shuffle(memory_buffer)
        states = np.array(list(memory_buffer[:, 0]), dtype=np.float)
        rewards = np.array(list(memory_buffer[:, 2]), dtype=np.float)
        next_states = np.array(list(memory_buffer[:, 3]), dtype=np.float)
        done = np.array(list(memory_buffer[:, 4]), dtype=bool)
        # Tape for the critic
        with tf.GradientTape() as critic_tape:
            # Compute the target and the MSE between the current prediction
            target = rewards + (1 - done.astype(int)) * gamma * critic_net(tf.expand_dims(next_states, -1))
            prediction = critic_net(tf.expand_dims(states, -1))
            objective = tf.math.square(prediction - target)
            grads = critic_tape.gradient(objective, critic_net.trainable_variables)
            critic_optimizer.apply_gradients(zip(grads, critic_net.trainable_variables))


class OverrideReward(gymnasium.wrappers.NormalizeReward):
    """
    Gymansium wrapper useful to update the reward function of the environment

    """

    def step(self, action):
        previous_observation = self.env.unwrapped.s
        desc = self.env.unwrapped.desc.tolist()
        observation, reward, terminated, truncated, info = self.env.step(action)
        row, col = s_unpack[observation]
        cell_type = desc[row][col]
        if cell_type == b"H":
            reward = -100
        elif observation == MAP_SIZE-1 * MAP_SIZE + MAP_SIZE-1:
            reward = 100
        else:
            # reward = -distances[previous_observation][action] / MAX_DIST
            reward = 0
        return observation, reward, terminated, truncated, info


def main():
    print("\n***************************************************")
    print("*  Welcome to the eleventh lesson of the RL-Lab!  *")
    print("*                 (DRL in Practice)               *")
    print("***************************************************\n")

    _training_steps = 5000

    # Crete the environment and add the wrapper for the custom reward function
    # env = gymnasium.make("FrozenLake-v1", max_episode_steps=100, map_name=f"{MAP_SIZE}x{MAP_SIZE}", is_slippery=False,
    #                      render_mode="human")
    env = gymnasium.make("FrozenLake-v1", max_episode_steps=100, map_name=f"{MAP_SIZE}x{MAP_SIZE}", is_slippery=False)
    env = OverrideReward(env)

    # Create the networks and perform the actual training
    actor_net = createDNN(1, 4, nLayer=2, nNodes=32, last_activation="softmax")
    critic_net = createDNN(1, 1, nLayer=2, nNodes=32, last_activation="linear")
    rewards_training, ep_lengths = training_loop(env, actor_net, critic_net, A2C, frequency=10,
                                                 episodes=_training_steps)

    # Save the trained neural network
    actor_net.save("FrozenLakeActor.h5")

    # Plot the results
    t = np.arange(0, _training_steps, 10)
    plt.plot(t, ep_lengths, label="A2C", linewidth=3)
    plt.xlabel("epsiodes", fontsize=16)
    plt.ylabel("success", fontsize=16)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
