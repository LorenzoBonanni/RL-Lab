import os, sys, numpy

import numpy as np

from tools.DangerousGridWorld import GridWorld


def on_policy_mc(environment: GridWorld, maxiters=5000, eps=0.3, gamma=0.99 ):
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
	p_size = (environment.observation_space, environment.action_space)
	p = np.full(p_size, eps/environment.action_space)
	idx = np.random.randint(0, environment.action_space-1, environment.observation_space)
	p[np.arange(environment.observation_space), idx] = (1 - eps) + eps / environment.action_space

	Q = np.array([[0 for _ in range(environment.action_space)] for _ in range(environment.observation_space)])
	returns = np.array([[0 for _ in range(environment.action_space)] for _ in range(environment.observation_space)])
	for _ in range(maxiters):
		episode = environment.sample_episode(policy=p)
		G = 0
		for s, a, r in episode:
			G = (gamma * G) + r
			returns[s][a] = G
			Q[s][a] = np.mean(returns[s][a])
			a_star = np.argmax(Q[s])
			for action in range(environment.action_space):
				if action == a_star:
					p[s][a] = (1 - eps) + eps / environment.action_space
				else:
					p[s][a] = eps / environment.action_space
	deterministic_policy = [numpy.argmax(p[state]) for state in range(environment.observation_space)]	
	return deterministic_policy


def main():
	print( "\n*************************************************" )
	print( "*  Welcome to the third lesson of the RL-Lab!   *" )
	print( "*       (Temporal Difference Methods)           *" )
	print( "**************************************************" )

	print("\nEnvironment Render:")
	env = GridWorld()
	env.render()

	print( "\n3) MC On-Policy" )
	mc_policy = on_policy_mc( env )
	env.render_policy( mc_policy )
	print( "\tExpected reward following this policy:", env.evaluate_policy(mc_policy) )
	

if __name__ == "__main__":
	main()
