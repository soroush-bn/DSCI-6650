from grid import Grid
import numpy as np
import matplotlib.pyplot as plt
from utils import *

import argparse

def main():
    parser = argparse.ArgumentParser(description="Simulate multi-armed bandit problem.")
    parser.add_argument('--theta', type=float, default=1e-3, help='theta threshold')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--gui', action='store_true' )
    parser.add_argument('-q1', '--question1', action='store_true')
    parser.add_argument('-q2', '--question2', action='store_true')
    parser.add_argument('-t', '--tune', action='store_true')

    args = parser.parse_args()
    
    grid =Grid()
    gamma = args.gamma
    theta=args.theta
    gui=args.gui
    q1 =args.question1
    q2= args.question2
    tune = args.tune

    if q1:
        v_bellman_eq = value_function_with_solving_bellman_eq(grid,gamma)
        print_details(v_bellman_eq,value_function_with_solving_bellman_eq.__name__,grid)
        v_bellman_eq= np.array(v_bellman_eq)
        v_bellman_eq=v_bellman_eq.reshape(grid.shape).tolist()
        
        grid =Grid()
        v_policy_evaluation = value_function_with_iterative_policy_evaluation(grid,gamma)
        # arr = np.array(v)
        print_details(v_policy_evaluation,value_function_with_iterative_policy_evaluation.__name__,grid)

        if gui:
            plot_values([v_bellman_eq,v_policy_evaluation],["values of bellman eq","values of policy evaluation"])

        if tune : 
            discount_factors = [0.95,0.75,0.50,0]
            v_bellmans = []
            v_policy_evals =[]
            for df in discount_factors:
                v_bellman_eq = value_function_with_solving_bellman_eq(grid,df)
                v_policy_evaluation = value_function_with_iterative_policy_evaluation(grid,df)
                v_bellmans.append(v_bellman_eq)
                v_policy_evals.append(v_policy_evaluation)


            plot_values(v_policy_evals,["gamma=0.95","gamma=0.70","gamma=0.50","gamma=0"])


    if q2:


        v_bellman_eq = value_function_with_solving_bellman_eq(grid,gamma)
        print("*" *12)
        print("optimal policy solving bellamn eq: \n")
        v_bellman_eq= np.array(v_bellman_eq)
        v_bellman_eq=v_bellman_eq.reshape(grid.shape).tolist()
        p_bellman= arg_max_from_value_function(v_bellman_eq,grid)
        pprint(p_bellman)
        print("--"*10)

        print("optimal policy with policy iteration: \n")
        grid=Grid()
        v_policy_iteration,p_policy_iteration,_ = policy_iteration(grid,discount_factor=0.95)
        pprint(p_policy_iteration)
        print("--"*10)

        print("optimal policy with value iteration: \n")
        grid=Grid()
        v_value_iteration,p_value_iteration,_ = policy_improvement_with_value_iteration(grid)
        pprint(p_value_iteration)
        print("--"*10)

        if gui:
            
            plot_policies([p_bellman,p_policy_iteration,p_value_iteration],["bellman eq","policy_iteration","value_iteration"],grid)

        if tune: 
            discount_factors = [0.95,0.70,0.50,0]
            avg_rewards_over_dfs_policy_iteration = []
            avg_rewards_over_dfs_value_iteration = [] 
            
            for df in discount_factors : 
                grid=Grid()
                v_policy_iteration,p_policy_iteration,avg_rewards_policy_iteration = policy_iteration(grid,discount_factor=df,playing=True)
                
                grid=Grid()
                v_value_iteration,p_value_iteration, avg_rewards_value_iteration = policy_improvement_with_value_iteration(grid,discount_factor=df,playing=True)

                avg_rewards_over_dfs_value_iteration.append(play(p_value_iteration))
                avg_rewards_over_dfs_policy_iteration.append(play(p_policy_iteration))

            print(avg_rewards_over_dfs_value_iteration)
            print(avg_rewards_over_dfs_policy_iteration)
            



def create_transition_and_reward_matrices(grid):
    shape = grid.shape
    num_states = shape[0] * shape[1]
    P = np.zeros((num_states, num_states))
    R = np.zeros(num_states)
    
    states = get_all_states(shape)
    state_to_index = {state: idx for idx, state in enumerate(states)}
    
    for state in states:
        state_idx = state_to_index[state]
        grid.current_state = state
        for action in grid.action_set:
            grid.current_state=state # for starting from the same state
            next_state, reward = grid.move( action)
            next_state_idx = state_to_index[next_state]
            P[state_idx, next_state_idx] += 0.25  
            R[state_idx] += 0.25 * reward  

    return P, R

def value_function_with_solving_bellman_eq(grid, discount_factor=0.95):
    P, R = create_transition_and_reward_matrices(grid)
    num_states = P.shape[0]
    I = np.eye(num_states)
    V = np.linalg.solve(I - discount_factor * P, R)
    return V




def value_function_with_iterative_policy_evaluation(grid, discount_factor=0.95, theta=1e-3,policy_prob=None):
    shape = grid.shape
    value_function = np.zeros(shape)
    model_transition_prob = 1 
    if policy_prob==None:
        policy_prob = [[0]*grid.shape[0]]*grid.shape[1]
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                policy_prob[i][j]= {"left":0.25,"right":0.25,"up":0.25,"down": 0.25}# Equal probability for each action
                policy = arg_max_from_value_function(value_function,grid)
    while True:
        delta = 0
        # for state in states:
        for i in range(shape[0]):
            for j in range(shape[1]):
                grid.current_state= (i,j)
                v = value_function[i][j]
                new_value = 0
                state = grid.current_state
                for action in grid.action_set:
                    #todo
                    grid.current_state=state # for starting from the same state
                    next_state, reward = grid.move(action) # we dont need P (transitions probabilities are all 1 execpt from special cells that you know for sure you are on that state)???
                    new_value += policy_prob[i][j][action] * (reward + discount_factor * value_function[next_state[0]][next_state[1]])
                value_function[i][j] = new_value
                delta = max(delta, abs(v - new_value))
        if delta < theta:
            break

    return value_function

def policy_iteration(grid,discount_factor = 0.95,playing=False):
    # init 
    v = np.zeros(grid.shape)
    policy_prob = [[0]*grid.shape[0]]*grid.shape[1]
    policy = [["right"]*grid.shape[1] for _ in range(grid.shape[0])]
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            policy_prob[i][j]= {"left":0.25,"right":0.25,"up":0.25,"down": 0.25}# Equal probability for each action

    avg_rewards = [] 
    ### does argmax means the policy(s) should only return one action at a time or it can also return a probability set ??
    while True : 
        #policy evaluation 
        if playing: avg_rewards.append(play(policy))
        policy_prob = deterministic_policy_to_policy_probs(policy,grid)
        value_function = value_function_with_iterative_policy_evaluation(grid,policy_prob=policy_prob)
        # policy improvement 
        policy_stable = True
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                old_action = max(policy_prob[i][j], key=policy_prob[i][j].get)
                policy = arg_max_from_value_function(value_function,grid)
                if old_action != max(policy_prob[i][j], key=policy_prob[i][j].get):
                    policy_stable = False
                if policy_stable : 
                    return value_function,policy,avg_rewards
    


            



def policy_improvement_with_value_iteration(grid, discount_factor=0.95, theta=1e-3,playing=False):
    shape = grid.shape
    value_function = np.zeros(shape)
    avg_rewards = [] 
    while True:
        delta = 0 
        for i in range(shape[0]):
            for j in range(shape[1]):
                grid.current_state= (i,j)
                v = value_function[i][j]
                new_value = -9999
                state = grid.current_state
                for action in grid.action_set:
                    grid.current_state=state # for starting from the same state
                    next_state,reward = grid.move(action)
                    new_value = max(new_value,(reward + discount_factor * value_function[next_state[0]][next_state[1]]) )
                delta = max(delta, abs(v - new_value))
                value_function[state] = new_value

            if delta < theta:
                break
        if playing:
            policy = arg_max_from_value_function(value_function,grid)
            avg_rewards.append(play(policy))
        if delta < theta:
                break
    policy = arg_max_from_value_function(value_function,grid)

    return value_function,policy,avg_rewards


                

def play(policy,n =100,steps=100):

    rewards = []
    
    for i in range(n):
        grid= Grid()
        terminal= False
        reward = 0 
        for _ in range(steps): 
            state = grid.current_state
            action = policy[state[0]][state[1]]
            next_state, r = grid.move(action)
            reward += r
        rewards.append(reward)

    return sum(rewards)/n



if __name__=="__main__":
    main()