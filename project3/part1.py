import argparse
from grid import Grid
import numpy as np
import random
from utils import *
def epsilon_greedy_policy(state, Q, epsilon, n_actions=4):
    action_probs = np.ones(n_actions) * (epsilon / n_actions)
    action_map = {'left': 0, 'right':1,'up':2,'down':3}
    greedy_action = max(Q[state[0]][state[1]],key=Q[state[0]][state[1]].get )
    greedy_action = action_map[greedy_action]
    action_probs[greedy_action] += (1.0 - epsilon)
    return action_probs

def get_action(state, Q, epsilon, n_actions=4):
    action_probs = epsilon_greedy_policy(state, Q, epsilon, n_actions)
    action = np.random.choice(np.arange(n_actions), p=action_probs)
    return action[0]

def epsilon_greedy_policy2(state,Q1,Q2,epsilon,n_actions=4):
    action_probs = np.ones(n_actions) * (epsilon / n_actions)
    action_map = {'left': 0, 'right':1,'up':2,'down':3}
    greedy_action = max(Q1[state[0]][state[1]]+Q2[state[0]][state[1]],key=Q1[state[0]][state[1]].get )
    greedy_action = action_map[greedy_action]
    action_probs[greedy_action] += (1.0 - epsilon)
    return action_probs

#todo Sarsa
def sarsa(grid: Grid,n_episodes = 10000, alpha=0.1,epsilon=0.1,discount=0.95,eps_decay=0.00005):
    print("starting SARSA \n")
    print("from state: " + str(grid.current_state))
    # policy_prob = [[{"left": 0.25, "right": 0.25, "up": 0.25, "down": 0.25} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    # policy = [["right" for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    steps=[]
    sum_of_rewards = []

    Q = [[{"left": 10, "right": 10, "up": 10, "down": 10} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    # Returns = [[{"left": [], "right": [], "up": [], "down": []} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    # rewards = []
    
    for i in range(n_episodes):
        grid.current_state= grid.blue_pos

        if epsilon > 0.01:
            epsilon -= eps_decay
        if i% 100 ==0 :
            print("elapsed: %"+ str(i/n_episodes *100))  
        terminal = False
        S = grid.current_state
        action_probs = epsilon_greedy_policy(S,Q,epsilon)
        A = random.choices((grid.action_set), (action_probs.tolist()),k=1)[0]
        step_counter=0
        reward = 0 
        while not terminal: #or replace with steps ?
            step_counter+=1
            # if step_counter%10000 ==0 :
            #     print(A)
            #     print(action_probs)
            #     print(Q[S[0]][S[1]])

            S_prime,R,terminal = grid.move(A)
            reward+=R
            action_probs = epsilon_greedy_policy(S_prime,Q,epsilon)
            A_prime = random.choices(list(grid.action_set),list(action_probs),k=1)[0]
            Q[S[0]][S[1]][A] = Q[S[0]][S[1]][A] + alpha*(R + discount* Q[S_prime[0]][S_prime[1]][A_prime] - Q[S[0]][S[1]][A])
            S = S_prime
            A = A_prime
        steps.append(step_counter)
        sum_of_rewards.append(reward)
    return Q,steps,sum_of_rewards
    


# todo Q Learning
def Qlearning(grid: Grid,n_episodes = 10000, alpha=0.1,epsilon=0.05,discount=0.95,eps_decay=0.00005):
    # policy_prob = [[{"left": 0.25, "right": 0.25, "up": 0.25, "down": 0.25} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    # policy = [["right" for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]

    Q = [[{"left": 10, "right": 10, "up": 10, "down": 10} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    # Returns = [[{"left": [], "right": [], "up": [], "down": []} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    # rewards = []
    steps=[]
    sum_of_rewards = []

    for i in range(n_episodes):
        grid.current_state= grid.blue_pos

        terminal = False
        S = grid.current_state
        if epsilon > 0.01:
            epsilon -= eps_decay
        if i% 100 ==0 :
            print("elapsed: %"+ str(i/n_episodes *100))  
        step_counter=0
        reward=0
        while not terminal: #or replace with steps ?
            step_counter+=1

            action_probs = epsilon_greedy_policy(S,Q,epsilon)
            A = random.choices(grid.action_set,action_probs,k=1)[0]
            # action_probs = epsilon_greedy_policy(S_prime,Q,epsilon)
            # A_prime = random.choice(grid.action_set,p = action_probs)
            S_prime,R,terminal = grid.move(A)
            reward+=R
            Q[S[0]][S[1]][A] = Q[S[0]][S[1]][A] + alpha*(R + discount* max(Q[S_prime[0]][S_prime[1]].values()) - Q[S[0]][S[1]][A])
            S = S_prime
            # A = A_prime
        steps.append(step_counter)
        sum_of_rewards.append(reward)
    return Q,steps,sum_of_rewards

# todo expected sarsa
def expected_sarsa(grid: Grid,n_episodes = 1000, alpha=0.1,epsilon=0.05,discount=0.95,eps_decay=0.00005):
    # policy_prob = [[{"left": 0.25, "right": 0.25, "up": 0.25, "down": 0.25} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    # policy = [["right" for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]

    Q = [[{"left": 0, "right": 0, "up": 0, "down": 0} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    # Returns = [[{"left": [], "right": [], "up": [], "down": []} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    # rewards = []
    steps=[]
    sum_of_rewards = []
    
    for i in range(n_episodes):
        grid.current_state= grid.blue_pos
        if epsilon > 0.01:
            epsilon -= eps_decay
        if i% 100 ==0 :
            print("elapsed: %"+ str(i/n_episodes *100))  
        terminal = False
        S = grid.current_state
        action_probs = epsilon_greedy_policy(S,Q,epsilon)
        A = random.choices(grid.action_set,action_probs,k=1)[0]
        step_counter=0
        reward = 0 
        while not terminal: #or replace with steps ?
            step_counter+=1

            S_prime,R,terminal = grid.move(A)
            action_probs = epsilon_greedy_policy(S_prime,Q,epsilon)
            A_prime = random.choices(grid.action_set,action_probs,k=1)[0]
            expected_target = 0 
            for a in grid.action_set:
                expected_target+= action_probs[a] * Q[S_prime[0]][S_prime[S[1]]][a]
            Q[S[0]][S[1]][A] = Q[S[0]][S[1]][A] + alpha*(R + discount* expected_target  - Q[S[0]][S[1]][A])
            S = S_prime
            A = A_prime
            reward+=R
        steps.append(step_counter)
        sum_of_rewards.append(reward)
    return Q,steps,sum_of_rewards

# todo d learning
def Doublelearning(grid: Grid,n_episodes = 1000, alpha=0.1,epsilon=0.05,discount=0.95,eps_decay=0.00005):
    # policy_prob = [[{"left": 0.25, "right": 0.25, "up": 0.25, "down": 0.25} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    # policy = [["right" for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]

    Q1 = [[{"left": 0, "right": 0, "up": 0, "down": 0} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    Q2 = [[{"left": 0, "right": 0, "up": 0, "down": 0} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    steps=[]
    sum_of_rewards = []
    # Returns = [[{"left": [], "right": [], "up": [], "down": []} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    # rewards = []

    for i in range(n_episodes):
        grid.current_state= grid.blue_pos
        if epsilon > 0.01:
            epsilon -= eps_decay
        if i% 100 ==0 :
            print("elapsed: %"+ str(i/n_episodes *100))  
        terminal = False
        S = grid.current_state
        step_counter= 0
        reward=0
        while not terminal: #or replace with steps ?
            step_counter+=1
            action_probs = epsilon_greedy_policy2(S,Q1, Q2,epsilon)
            A = random.choices(grid.action_set, action_probs,k=1)[0]
            # action_probs = epsilon_greedy_policy(S_prime,Q,epsilon)
            # A_prime = random.choice(grid.action_set,p = action_probs)
            S_prime,R,terminal = grid.move(A)
            reward+=R
            if np.random.binomial(1, 0.5, 1)==0:
                
                Q1[S[0]][S[1]][A] = Q1[S[0]][S[1]][A] + alpha*(R + discount* max(Q2[S_prime[0]][S_prime[1]].values()) - Q1[S[0]][S[1]][A])
            else:
                Q2[S[0]][S[1]][A] = Q2[S[0]][S[1]][A] + alpha*(R + discount* max(Q1[S_prime[0]][S_prime[1]].values()) - Q2[S[0]][S[1]][A])

            S = S_prime
            # A = A_prime
        steps.append(step_counter)
        sum_of_rewards.append(reward)
        
    return Q1,Q2,steps,sum_of_rewards

def main():
    parser = argparse.ArgumentParser(description="Simulate multi-armed bandit problem.")
    parser.add_argument('--episodes', type=int, default=4000, help='number of episodes')
    parser.add_argument('--epsilon', type=float, default=0.2, help='epsilon')
    parser.add_argument('--alpha', type=float, default=0.1, help='epsilon')

    parser.add_argument('--gui', action='store_true' )
    parser.add_argument('--tune', action='store_true' )

    args = parser.parse_args()
    
    gui=args.gui
    alpha=args.alpha
    tune = args.tune
    epsilon = args.epsilon
    n_episodes = args.episodes


    # ##Sarsa
    # grid = Grid()
    # Q_sarsa,steps_sarsa,rewards_sarsa= sarsa(grid,n_episodes,alpha,epsilon)
    # print("finished Sarsa: final q: ")
    # print(Q_sarsa)
    # # plot_state_values(Q_sarsa)
    # # plot_policy(grid,Q_sarsa)


    # ## Q learning
    # grid = Grid()
    # Q_Qlearning,steps_qlearning,rewards_qlearning = Qlearning(grid,n_episodes,alpha,epsilon)
    # print("finished Q learning: final Q: ")
    # print(Q_Qlearning)
    # # plot_state_values(Q_sarsa)
    # # plot_policy(grid,Q_Qlearning)

        ## Q learning
    grid = Grid()
    Q_ExpectedSarsa,steps_ExpectedSarsa,rewards_ExpectedSarsa = expected_sarsa(grid,n_episodes,alpha,epsilon)
    print("finished expected sarsa: final Q: ")
    print(Q_ExpectedSarsa)

    ## qqlearning
    grid = Grid()
    Q_DQlearning,steps_DQlearning,rewards_DQlearning = Doublelearning(grid,n_episodes,alpha,epsilon)
    print("finished expected sarsa: final Q: ")
    print(Q_DQlearning)

    plot_policies_grid(grid,[Q_ExpectedSarsa,Q_DQlearning],["eSARSA","dQLEARNING"])
    plot_time_steps([steps_ExpectedSarsa,steps_DQlearning],["eSARSA","dQLEARNING"])
    plot_rewards([rewards_ExpectedSarsa,rewards_DQlearning],["eSARSA","dQLEARNING"])




if __name__=="__main__":
    main()