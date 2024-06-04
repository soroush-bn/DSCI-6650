from bandit import Bandit
import argparse
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

def main():
    
    parser = argparse.ArgumentParser(description="Simulate multi-armed bandit problem.")
    parser.add_argument('--num_problems', type=int, default=1000, help='Number of problems to run')
    parser.add_argument('--steps', type=int, default=10000, help='Number of steps for each problem')
    parser.add_argument('--epsilon', type=float, default=0.2, help='Epsilon value for epsilon-greedy strategy')
    parser.add_argument('--alpha', type=float, default=0.1, help='Learning rate for gradient bandit strategy')
    parser.add_argument('--save', type=bool, default=False ,help='Save plots in SAVE_DIR PATH' )
    parser.add_argument('--non_stationary', type=str, default="abrupt" ,help='activate non-stationary bandit, value can be gradual or abrupt' )
    parser.add_argument('--gradual_type', type=str, default="revert" ,help='type of gradual change(if any), value can be drift or revert' )
    args = parser.parse_args()

    num_problems = args.num_problems
    steps = args.steps
    epsilon = args.epsilon
    alpha = args.alpha
    non_stationary = args.non_stationary
    gradual_type = args.gradual_type
    seed = 42
    optimistic_greedy_rewards = np.zeros(steps)
    e_greedy_fixed_rewards = np.zeros(steps)
    e_greedy_decreasing_rewards = np.zeros(steps)
    optimistic_greedy_average_rewards= np.zeros(num_problems)
    e_greedy_fixed_average_rewards= np.zeros(num_problems)
    e_greedy_decreasing_average_rewards= np.zeros(num_problems)
    bandit = Bandit()
    print("--"*10)
    print(f"Part2: Starting bandit problem with following params: ")
    print(f"number of steps:\t{steps}\n number of problems:\t{num_problems}\n epsilon:\t{epsilon}\n alpha:\t{alpha}\n  non stationary type:\t{non_stationary}\n gradual type:\t{gradual_type}"
              )
    print("--"*10)
    #Optimistic greedy method 
    for i in range(num_problems):
        seed+=1
        np.random.seed(seed)
        optimistic_greedy_rewards , _ = bandit.run(steps=steps, init_q_estimate_type="optimistic",action_selection_type="greedy",epsilon=epsilon,non_stationary=non_stationary,gradual_type=gradual_type)
        optimistic_greedy_average_rewards[i] = np.mean(optimistic_greedy_rewards)
        e_greedy_fixed_rewards , _ = bandit.run(steps=steps, init_q_estimate_type="zero",action_selection_type="epsilon_greedy",epsilon=epsilon,non_stationary=non_stationary,gradual_type=gradual_type)
        e_greedy_fixed_average_rewards[i] = np.mean(e_greedy_fixed_rewards)
        e_greedy_decreasing_rewards , _ = bandit.run(steps=steps, init_q_estimate_type="zero",action_selection_type="epsilon_greedy",epsilon=epsilon,decreasing=True,non_stationary=non_stationary,gradual_type=gradual_type)
        e_greedy_decreasing_average_rewards[i] = np.mean(e_greedy_decreasing_rewards)


        
    # print(optimistic_greedy_average_rewards)
    # print(e_greedy_fixed_average_rewards)
    # print(e_greedy_decreasing_average_rewards)    
    
    data = [optimistic_greedy_average_rewards, e_greedy_fixed_average_rewards, e_greedy_decreasing_average_rewards]
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=['Optimistic Greedy', 'Epsilon-Greedy Fixed', 'Epsilon-Greedy Decreasing'])
    plt.ylabel('Average Reward')
    plt.title('Box Plot of Average Rewards')
    
    if args.save:
        plt.savefig('part2_10k_1k_abrupt_e2.png')
    else:
        plt.show()

    # plt.figure(figsize=(10, 6))
    # plt.violinplot(optimistic_greedy_average_rewards, showmeans=False, showmedians=True)
    # plt.violinplot(e_greedy_fixed_average_rewards, showmeans=False, showmedians=True)
    # plt.violinplot(e_greedy_decreasing_average_rewards, showmeans=False, showmedians=True)
    # plt.title('Reward Distributions for 10-arm Bandit')
    # plt.xlabel('Arm')
    # plt.ylabel('Reward Distribution')
    # plt.show()
    

    #epsilon greedy with a fixed step size

    #epsilon greedy with a decreasing step size



    

if __name__ == "__main__":
    main()