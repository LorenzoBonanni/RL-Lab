import numpy
import os
import sys

module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path: sys.path.append(module_path)
from DangerousGridWorld import GridWorld


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
    if numpy.random.random() < epsilon:
        return numpy.random.choice(q.shape[1])
    return q[state].argmax()


def q_learning(environment, episodes, alpha, gamma, expl_func, expl_param):
    """
    Performs the Q-Learning algorithm for a specific environment

    Args:
        environment: OpenAI Gym environment
        episodes: number of episodes for training
        alpha: alpha parameter
        gamma: gamma parameter
        expl_func: exploration function (epsilon_greedy, softmax)
        expl_param: exploration parameter (epsilon, T)

    Returns:
        (policy, rewards, lengths): final policy, rewards for each episode [array], length of each episode [array]
    """

    q = numpy.zeros((environment.observation_space, environment.action_space))  # Q(s, a)
    rews = numpy.zeros(episodes)
    lengths = numpy.zeros(episodes)
    for ep_n in range(episodes):
        s = environment.random_initial_state()
        terminal = environment.is_terminal(s)
        while not terminal:
            a = expl_func(q, s, expl_param)
            s_1 = environment.sample(a, s)
            r = environment.R[s_1]
            rews[ep_n] += r
            terminal = environment.is_terminal(s_1)

            q[s][a] += alpha * (r + gamma * numpy.max(q[s_1][a] - q[s][a]))
            s = s_1
            lengths[ep_n]+=1

    policy = q.argmax(axis=1)  # q.argmax(axis=1) automatically extract the policy from the q table
    return policy, rews, lengths


def sarsa(environment, episodes, alpha, gamma, expl_func, expl_param):
    """
    Performs the SARSA algorithm for a specific environment

    Args:
        environment: OpenAI gym environment
        episodes: number of episodes for training
        alpha: alpha parameter
        gamma: gamma parameter
        expl_func: exploration function (epsilon_greedy, softmax)
        expl_param: exploration parameter (epsilon, T)

    Returns:
        (policy, rewards, lengths): final policy, rewards for each episode [array], length of each episode [array]
    """

    q = numpy.zeros((environment.observation_space, environment.action_space))  # Q(s, a)
    rews = numpy.zeros(episodes)
    lengths = numpy.zeros(episodes)

    for i in range(episodes):
        curr_state = environment.random_initial_state()
        action = expl_func(q, curr_state, expl_param)
        done = False
        while not done:
            next_state = environment.sample(action, curr_state)
            reward = environment.R[next_state]
            done = environment.is_terminal(next_state)
            next_a = expl_func(q, next_state, expl_param)
            q[curr_state, action] += alpha * (reward + (gamma * q[next_state, next_a]) - q[curr_state, action])
            curr_state = next_state
            action = next_a
            lengths[i] += 1
            rews[i] += reward

    policy = q.argmax(axis=1)  # q.argmax(axis=1) automatically extract the policy from the q table
    return policy, rews, lengths


def main():
    print("\n*************************************************")
    print("*  Welcome to the fourth lesson of the RL-Lab!   *")
    print("*        (Temporal Difference Methods)           *")
    print("**************************************************")

    print("\nEnvironment Render:")
    env = GridWorld()
    env.render()

    # Learning parameters
    episodes = 500
    alpha = .3
    gamma = .9
    epsilon = .1

    # Executing the algorithms
    policy_qlearning, rewards_qlearning, lengths_qlearning = q_learning(env, episodes, alpha, gamma, epsilon_greedy,
                                                                        epsilon)
    policy_sarsa, rewards_sarsa, lengths_sarsa = sarsa(env, episodes, alpha, gamma, epsilon_greedy, epsilon)

    print("\n4) Q-Learning")
    env.render_policy(policy_qlearning)
    print("\tExpected reward training with Q-Learning:", numpy.round(numpy.mean(rewards_qlearning), 2))
    print("\tAverage steps training with Q-Learning:", numpy.round(numpy.mean(lengths_qlearning), 2))

    print("\n5) SARSA")
    env.render_policy(policy_sarsa)
    print("\tExpected reward training with SARSA:", numpy.round(numpy.mean(rewards_sarsa), 2))
    print("\tAverage steps training with SARSA:", numpy.round(numpy.mean(lengths_sarsa), 2))


if __name__ == "__main__":
    main()
