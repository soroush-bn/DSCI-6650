from bandit import Bandit
import argparse
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def main():
    parser = argparse.ArgumentParser(description="Simulate multi-armed bandit problem.")
    parser.add_argument('--num_problems', type=int, default=1000, help='Number of problems to run')
    parser.add_argument('--steps', type=int, default=1000, help='Number of steps for each problem')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon value for epsilon-greedy strategy')
    parser.add_argument('--alpha', type=float, default=0.1, help='Learning rate for gradient bandit strategy')
    parser.add_argument('--save', type=bool, default=False ,help='Save plots in SAVE_DIR PATH' )
    
    args = parser.parse_args()

    num_problems = args.num_problems
    steps = args.steps
    epsilon = args.epsilon
    alpha = args.alpha

    avg_rewards_greedy = np.zeros(steps)
    avg_rewards_epsilon_greedy = np.zeros(steps)
    avg_rewards_optimistic = np.zeros(steps)
    avg_rewards_gradient = np.zeros(steps)

    optimal_action_percentage_greedy = np.zeros(steps)
    optimal_action_percentage_epsilon_greedy = np.zeros(steps)
    optimal_action_percentage_optimistic = np.zeros(steps)
    optimal_action_percentage_gradient = np.zeros(steps)
    print("--"*10)
    print(f"Part1: Starting bandit problem with following params: ")
    print(f"number of steps:\t{steps}\n \
              number of problems:\t{num_problems}\n \
              epsilon:\t{epsilon}\n \
              alpha:\t{alpha}\n
              "
              
              )
    print("--"*10)
    for i in range(num_problems):
        bandit = Bandit()
        rewards, optimal_actions = bandit.run(steps=steps, init_q_estimate_type="zero", action_selection_type="greedy")
        avg_rewards_greedy += rewards
        optimal_action_percentage_greedy += optimal_actions

        rewards, optimal_actions = bandit.run(steps=steps, init_q_estimate_type="zero", action_selection_type="epsilon_greedy", epsilon=epsilon)
        avg_rewards_epsilon_greedy += rewards
        optimal_action_percentage_epsilon_greedy += optimal_actions

        rewards, optimal_actions = bandit.run(steps=steps, init_q_estimate_type="optimistic", action_selection_type="greedy")
        avg_rewards_optimistic += rewards
        optimal_action_percentage_optimistic += optimal_actions

        rewards, optimal_actions = bandit.run(steps=steps, init_q_estimate_type="zero", action_selection_type="gradient", epsilon=epsilon)
        avg_rewards_gradient += rewards
        optimal_action_percentage_gradient += optimal_actions

    avg_rewards_greedy /= num_problems
    avg_rewards_epsilon_greedy /= num_problems
    avg_rewards_optimistic /= num_problems
    avg_rewards_gradient /= num_problems

    optimal_action_percentage_greedy /= num_problems
    optimal_action_percentage_epsilon_greedy /= num_problems
    optimal_action_percentage_optimistic /= num_problems
    optimal_action_percentage_gradient /= num_problems

    plt.figure(figsize=(10, 6))
    plt.plot(avg_rewards_greedy, label="Greedy")
    plt.plot(avg_rewards_epsilon_greedy, label="Epsilon-Greedy")
    plt.plot(avg_rewards_optimistic, label="Optimistic")
    plt.plot(avg_rewards_gradient, label="Gradient Bandit")
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs. Steps for Different Bandit Strategies')
    plt.legend()
    if(args.save):
        plt.savefig("avg_reward1k.png")
    plt.show()



    plt.figure(figsize=(10, 6))
    plt.plot(optimal_action_percentage_greedy, label="Greedy")
    plt.plot(optimal_action_percentage_epsilon_greedy, label="Epsilon-Greedy")
    plt.plot(optimal_action_percentage_optimistic, label="Optimistic")
    plt.plot(optimal_action_percentage_gradient, label="Gradient Bandit")
    plt.xlabel('Steps')
    plt.ylabel('Optimal Action Percentage')
    plt.title('Optimal Action Percentage vs. Steps for Different Bandit Strategies')
    plt.legend()
    if(args.save):
        plt.savefig("optimal_action1k.png")
    plt.show()




if __name__ == "__main__":
    # Bandit().pilot_run()
    main()
