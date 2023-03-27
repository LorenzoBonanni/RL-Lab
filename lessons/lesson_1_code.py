import random
from typing import Optional, Union, List, Tuple

import gym
import numpy
from gym.core import RenderFrame, ActType, ObsType

from tools.DangerousGridWorld import GridWorld


def random_dangerous_grid_world(environment):
    """
	Performs a random trajectory on the given Dangerous Grid World environment 
	
	Args:
		environment: OpenAI Gym environment
		
	Returns:
		trajectory: an array containing the sequence of states visited by the agent
	"""

    trajectory = []
    start_position = 0
    trajectory.append(start_position)
    for _ in range(10):
        new_state = environment.sample(random.choice(
            range(environment.action_space)),
            start_position
        )
        trajectory.append(new_state)
        if environment.is_terminal(new_state):
            break

    return trajectory


class RecyclingRobot(gym.Env):
    """
    Class that implements the environment Recycling Robot of the book: 'Reinforcement
    Learning: an introduction, Sutton & Barto'. Example 3.3 page 52 (second edition).

    Attributes
    ----------
        observation_space : int
            define the number of possible actions of the environment
        action_space: int
            define the number of possible states of the environment
        actions: dict
            a dictionary that translate the 'action code' in human languages
        states: dict
            a dictionary that translate the 'state code' in human languages

    Methods
    -------
        reset( self )
            method that reset the environment to an initial state; returns the state
        step( self, action )
            method that perform the action given in input, computes the next state and the reward; returns
            next_state and reward
        render( self )
            method that print the internal state of the environment
    """
    def __init__(self):
        # Loading the default parameters
        self.alfa = 0.7
        self.beta = 0.7
        self.r_search = 0.5
        self.r_wait = 0.2

        # Defining the environment variables
        self.observation_space = 2
        self.action_space = 3
        self.actions = {0: 'SEARCH', 1: 'WAIT', 2: 'RECHARGE'}
        self.states = {0: 'HIGH', 1: 'LOW'}
        self.state = None
        self.reset()

    def reset(self, **kwargs):
        self.state = random.choice(range(self.observation_space))
        return self.state

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        # the first key is the state and the second is the actions
        transition_table = {
            # high state
            0: {
                # search action
                0: {
                    "probabilities": [self.alfa, 1 - self.alfa],
                    "states": [0, 1],
                    "rewards": [self.r_search, self.r_search]
                },
                # wait action
                1: {
                    "probabilities": [1],
                    "states": [0],
                    "rewards": [self.r_wait]
                },
                # recharge action
                2: {
                    "probabilities": [1],
                    "states": [0],
                    "rewards": [0]
                }
            },
            # low state
            1: {
                # search action
                0: {
                    "probabilities": [1 - self.beta, self.beta],
                    "states": [0, 1],
                    "rewards": [-3, self.r_search]
                },
                # wait action
                1: {
                    "probabilities": [1],
                    "states": [1],
                    "rewards": [self.r_wait]
                },
                # recharge action
                2: {
                    "probabilities": [1],
                    "states": [0],
                    "rewards": [0]
                }
            }
        }

        probs, states, rewards = transition_table[self.state][action].values()
        state_idx = random.choices(
            population=range(len(states)),
            weights=probs
        )[0]
        self.state = states[state_idx]
        reward = rewards[state_idx]
        return self.state, reward, False, None

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        print(self.state)


def main():
    print("\n************************************************")
    print("*  Welcome to the first lesson of the RL-Lab!  *")
    print("*             (MDP and Environments)           *")
    print("************************************************")

    print("\nA) Random Policy on Dangerous Grid World:")
    env = GridWorld()
    # env.render()
    random_trajectory = random_dangerous_grid_world(env)
    print("\nRandom trajectory generated:", random_trajectory)

    print("\nB) Custom Environment: Recycling Robot")
    env = RecyclingRobot()
    state = env.reset()
    ep_reward = 0
    for step in range(10):
        a = numpy.random.randint(0, env.action_space)
        new_state, r, _, _ = env.step(a)
        ep_reward += r
        print(
            f"\tFrom state '{env.states[state]}' selected action '{env.actions[a]}': \t total reward: {ep_reward:1.1f}")
        state = new_state


if __name__ == "__main__":
    main()