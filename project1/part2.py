from bandit import Bandit
import argparse
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

def main():
    
    parser = argparse.ArgumentParser(description="Simulate multi-armed bandit problem.")
    parser.add_argument('--num_problems', type=int, default=1000, help='Number of problems to run')
    parser.add_argument('--steps', type=int, default=10000, help='Number of steps for each problem')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon value for epsilon-greedy strategy')
    # parser.add_argument('--alpha', type=float, default=0.1, help='Learning rate for gradient bandit strategy')
    parser.add_argument('--save', type=bool, default=False ,help='Save plots in SAVE_DIR PATH' )
    
    args = parser.parse_args()

    num_problems = args.num_problems
    steps = args.steps
    epsilon = args.epsilon
    alpha = args.alpha
    seed = 42
    optimistic_greedy_rewards = np.zeros(steps)
    e_greedy_fixed_rewards = np.zeros(steps)
    e_greedy_decreasing_rewards = np.zeros(steps)
    bandit = Bandit()
    #Optimistic greedy method 
    for i in range(num_problems):
        seed+=1
        np.random.seed(seed)
        optimistic_greedy_rewards , _ = bandit.run(steps, "optimistic","greedy")
        e_greedy_fixed_rewards , _ = bandit.run(steps, "zero","epsilon_greedy")
        e_greedy_decreasing_rewards , _ = bandit.run(steps, "zero","epsilon_greedy",decreasing=True)


        
        
    

    #epsilon greedy with a fixed step size

    #epsilon greedy with a decreasing step size



    

if __name__ == "__main__":
    main()