import os
import sys

import numpy as np

module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path: sys.path.append(module_path)
from DangerousGridWorld import GridWorld


def value_iteration(environment, maxiters=300, discount=0.9, max_error=1e-3):
    """
    Performs the value iteration algorithm for a specific environment

    Args:
        environment: OpenAI Gym environment
        maxiters: timeout for the iterations
        discount: gamma value, the discount factor for the Bellman equation
        max_error: the maximum error allowd in the utility of any state

    Returns:
        policy: 1-d dimensional array of action identifiers where index `i` corresponds to state id `i`
    """

    U_1 = [0 for _ in range(environment.observation_space)]  # vector of utilities for states S
    delta = 0  # maximum change in the utility o any state in an iteration
    for _ in range(maxiters):
        U = U_1.copy()
        delta = 0  # maximum change in the utility o any state in an iteration
        for cs in range(environment.observation_space):
            values = [
                sum([
                    environment.transition_prob(cs, a, s1) * U[s1]
                    for s1 in range(environment.observation_space)
                ]) for a in range(len(environment.actions))
            ]
            if not environment.is_terminal(cs):
                U_1[cs] = environment.R[cs] + discount * max(values)
            else:
                U_1[cs] = environment.R[cs]

            delta = max(delta, abs(U_1[cs] - U[cs]))

        if delta <= max_error * (1 - discount) / discount:
            break

    return environment.values_to_policy(U)


def policy_iteration(environment, maxiters=300, discount=0.9, maxviter=10):
    """
    Performs the policy iteration algorithm for a specific environment

    Args:
        environment: OpenAI Gym environment
        maxiters: timeout for the iterations
        discount: gamma value, the discount factor for the Bellman equation
        maxviter: number of epsiodes for the policy evaluation

    Returns:
        policy: 1-d dimensional array of action identifiers where index `i` corresponds to state id `i`
    """

    p = [0 for _ in range(environment.observation_space)]  # initial policy
    U = [0 for _ in range(environment.observation_space)]  # utility array

    # 1) Policy Evaluation
    def policy_evaluation():
        for vi in range(maxviter):
            for s in range(len(U)):
                values = [
                    environment.transition_prob(s, p[s], s1) * U[s1]
                    for s1 in range(environment.observation_space)
                ]
                if not environment.is_terminal(s):
                    U[s] = environment.R[s] + discount * sum(values)
                else:
                    U[s] = environment.R[s]

    # 2) Policy Improvement
    for _ in range(maxiters):
        policy_evaluation()
        unchanged = True

        for cs in range(environment.observation_space):
            value1 = max([
                sum([
                    environment.transition_prob(cs, a, s1) * U[s1]
                    for s1 in range(environment.observation_space)
                ]) for a in range(len(environment.actions))
            ])

            value2 = sum([
                environment.transition_prob(cs, p[cs], s1) * U[s1]
                for s1 in range(environment.observation_space)
            ])

            if value1 > value2:
                p[cs] = np.argmax(
                    [
                        sum([
                            environment.transition_prob(cs, a, s1) * U[s1]
                            for s1 in range(environment.observation_space)
                        ]) for a in range(len(environment.actions))
                    ]
                )
                unchanged = False

        if unchanged:
            break

    return p


def main():
    print("\n************************************************")
    print("*  Welcome to the second lesson of the RL-Lab! *")
    print("*    (Policy Iteration and Value Iteration)    *")
    print("************************************************")

    print("\nEnvironment Render:")
    env = GridWorld()
    env.render()

    print("\n1) Value Iteration:")
    vi_policy = value_iteration(env)
    env.render_policy(vi_policy)
    print("\tExpected reward following this policy:", env.evaluate_policy(vi_policy))

    print("\n2) Policy Iteration:")
    pi_policy = policy_iteration(env)
    env.render_policy(pi_policy)
    print("\tExpected reward following this policy:", env.evaluate_policy(pi_policy))


if __name__ == "__main__":
    main()
