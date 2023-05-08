import warnings;

warnings.filterwarnings("ignore")
import os;

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf;
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gymnasium, collections


# Notice that the value function has only one output with a linear activation
# function in the last layer
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
    rewards_list, memory_buffer = [], collections.deque(maxlen=1000)
    averaged_rewards = []

    for ep in range(episodes):
        # reset the environment and obtain the initial state
        state = env.reset()[0]
        ep_reward = 0
        curr_traj = []
        while True:
            # select the action to perform
            p = actor_net(state.reshape(-1, 4)).numpy()[0]
            action = np.random.choice(2, p=p)

            # Perform the action, store the data in the memory buffer and update the reward
            next_state, reward, terminated, truncated, info = env.step(action)
            curr_traj.append([state, action, reward, next_state, terminated])
            ep_reward += reward

            # exit condition for the episode
            if terminated or truncated:
                break

            # update the current state
            state = next_state
        memory_buffer.append(curr_traj)
        # Perform the actual training
        if (ep + 1) % frequency == 0:
            updateRule(actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer)
            memory_buffer = []

        # Update the reward list to return
        rewards_list.append(ep_reward)
        averaged_rewards.append(np.mean(rewards_list))
        print(f"episode {ep:4d}: rw: {int(ep_reward):3d} (averaged: {np.mean(rewards_list):5.2f})")

    # Close the enviornment and return the rewards list
    env.close()
    return averaged_rewards


def A2C(actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer, gamma=0.99):
    """
    Main update rule for the A2C update. This function includes the updates for the actor network (or policy function)
    and for the critic network (or value function)

    """
    # update rule for the critic (value function)
    for _ in range(10):
        # Shuffle the memory buffer
        np.random.shuffle(memory_buffer)
        objectives = []
        # Tape for the critic
        with tf.GradientTape() as critic_tape:
            for trj in memory_buffer:
                trj = np.array(trj)
                states = np.array(list(trj[:, 0]), dtype=np.float)
                rewards = np.array(list(trj[:, 2]), dtype=np.float)
                actions = np.array(list(trj[:, 1]), dtype=int)
                next_states = np.array(list(trj[:, 3]), dtype=np.float)
                done = np.array(list(trj[:, 4]), dtype=bool)
                # Compute the target and the MSE between the current prediction
                target = rewards + (1 - done.astype(int)) * gamma * critic_net(next_states)
                prediction = critic_net(states)
                mse = tf.math.square(prediction - target)
                objectives.append(tf.math.reduce_mean(mse))
            # Perform the actual gradient-descent process
            objective = tf.math.reduce_mean(objectives)
            grads = critic_tape.gradient(objective, critic_net.trainable_variables)
            critic_optimizer.apply_gradients(zip(grads, critic_net.trainable_variables))

    # implement the update rule for the actor (policy function)
    # extract the information from the buffer for the policy update
    # Tape for the actor
    objectives = []
    with tf.GradientTape() as actor_tape:
        for trj in memory_buffer:
            trj = np.array(trj)
            states = np.array(list(trj[:, 0]), dtype=np.float)
            rewards = np.array(list(trj[:, 2]), dtype=np.float)
            actions = np.array(list(trj[:, 1]), dtype=int)
            next_states = np.array(list(trj[:, 3]), dtype=np.float)
            # compute the log-prob of the current trajectory and the objective function
            adv_a = rewards + gamma * critic_net(next_states).numpy().reshape(-1)
            adv_b = critic_net(states).numpy().reshape(-1)
            probs = actor_net(states)
            indices = tf.transpose(tf.stack([tf.range(probs.shape[0]), actions]))
            probs = tf.gather_nd(
                indices=indices,
                params=probs
            )
            objective = tf.math.log(probs) * adv_a - adv_b
            objectives.append(tf.reduce_mean(tf.reduce_sum(objective)))

        objective = - tf.math.reduce_mean(objectives)
        grads = actor_tape.gradient(objective, actor_net.trainable_variables)
        critic_optimizer.apply_gradients(zip(grads, actor_net.trainable_variables))


def main():
    print("\n*************************************************")
    print("*  Welcome to the tenth lesson of the RL-Lab!   *")
    print("*                    (A2C)                      *")
    print("*************************************************\n")

    _training_steps = 2500

    env = gymnasium.make("CartPole-v1")
    actor_net = createDNN(4, 2, nLayer=2, nNodes=32, last_activation="softmax")
    critic_net = createDNN(4, 1, nLayer=2, nNodes=32, last_activation="linear")
    rewards_naive = training_loop(env, actor_net, critic_net, A2C, episodes=_training_steps)

    t = np.arange(0, _training_steps)
    plt.plot(t, rewards_naive, label="A2C", linewidth=3)
    plt.xlabel("epsiodes", fontsize=16)
    plt.ylabel("reward", fontsize=16)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
