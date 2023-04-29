import numpy
import numpy as np

from tools.DangerousGridWorld import GridWorld


def on_policy_mc(environment, maxiters=5000, eps=0.3, gamma=0.99):
    """
    Performs the on policy version of the every-visit MC control

    Args:
        environment: OpenAI Gym environment
        maxiters: timeout for the iterations
        eps: random value for the eps-greedy policy (probability of random action)
        gamma: gamma value, the discount factor for the Bellman equation

    Returns:
        policy: 1-d dimensional array of action identifiers where index `i` corresponds to state id `i`
    """

    p = [[eps / environment.action_space for _ in range(environment.action_space)] for _ in
         range(environment.observation_space)]
    Q = [[0 for _ in range(environment.action_space)] for _ in range(environment.observation_space)]
    returns = [[[] for _ in range(environment.action_space)] for _ in range(environment.observation_space)]

    for _ in range(maxiters):
        episode = environment.sample_episode(p)
        g = 0

        for current_state, current_action, current_reward in reversed(episode):
            g = g * gamma + current_reward
            returns[current_state][current_action].append(g)
            Q[current_state][current_action] = sum(returns[current_state][current_action]) / len(
                returns[current_state][current_action])
            optimal_action = np.argmax(Q[current_state])
            for action in range(environment.action_space):
                if action == optimal_action:
                    p[current_state][action] = 1 - eps + eps / environment.action_space
                else:
                    p[current_state][action] = eps / environment.action_space

    deterministic_policy = [numpy.argmax(p[state]) for state in range(environment.observation_space)]
    return deterministic_policy


def main():
    print("\n*************************************************")
    print("*  Welcome to the third lesson of the RL-Lab!   *")
    print("*       (Temporal Difference Methods)           *")
    print("**************************************************")

    print("\nEnvironment Render:")
    env = GridWorld()
    env.render()

    print("\n3) MC On-Policy")
    mc_policy = on_policy_mc(env)
    env.render_policy(mc_policy)
    print("\tExpected reward following this policy:", env.evaluate_policy(mc_policy))


if __name__ == "__main__":
    main()
