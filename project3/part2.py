import argparse
from grid_part2 import Grid
import random
import numpy as np 

from utils import plot_values,plot_values_per_state

def generate_episode(grid: Grid, S0, policy_probs):
    # Generate an episode: (S0, A0, R1, ..., St-1, At-1, Rt)
    grid.current_state = S0
    history = []
    terminal = False
    action = choose_action(policy_probs,S0)
    while not terminal:
        pre_s = grid.current_state
        pre_a = action
        next_state, reward, terminal = grid.move(action)
        history.append((pre_s, pre_a, reward))
        action = choose_action(policy_probs, next_state)
    
    return history

def choose_action(policy_prob, state):
    i, j = state
    actions = list(policy_prob[i][j].keys())
    probabilities = list(policy_prob[i][j].values())
    chosen_action = random.choices(actions, probabilities)[0]
    return chosen_action

def get_feature_vector(state): #onehot encoding
    vector = np.zeros(49)  
    vector[state[0] * 7 + state[1]] = 1
    return vector

def get_value_function(state,w):
    x = get_feature_vector(state)
    return np.dot(w, x)

def compute_grid_value_function(grid, w):
    value_grid = np.zeros(grid.shape)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            state = (i, j)
            value_grid[i, j] = get_value_function(state, w)
    return value_grid


def gradient_MC(grid: Grid, n_episodes = 1000, alpha=0.1,discount=0.95): 
    policy_prob = [[{"left": 0.25, "right": 0.25, "up": 0.25, "down": 0.25} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])] #this is redundant and all this can be done by a random choice

    w = np.zeros(49) 
    for i in range(n_episodes):
        if i% 100 ==0 :
            print("elapsed: %"+ str(i/n_episodes *100))  
        state = grid.reset()
        history = generate_episode(grid, state,policy_prob)
        G = 0
        for t in range(len(history) - 1, -1, -1):
            next_state, action, reward = history[t]
            G = discount * G + reward
            x = get_feature_vector(next_state)
            w += alpha * (G - np.dot(w, x)) * x

    return w

def semi_gradient_TD(grid: Grid, n_episodes = 1000, alpha=0.1,discount=0.95): 
    w = np.zeros(49) 
    policy_prob = [[{"left": 0.25, "right": 0.25, "up": 0.25, "down": 0.25} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])] #this is redundant and all this can be done by a random choice

    for i in range(n_episodes):
        if i% 100 ==0 :
            print("elapsed: %"+ str(i/n_episodes *100))  
        state = grid.reset()
        while True:
            action = choose_action(policy_prob,state)
            next_state, reward, done = grid.move(action)
            delta = reward + discount * np.dot(w, get_feature_vector(next_state)) - np.dot(w, get_feature_vector(state))
            w += alpha * delta * get_feature_vector(state)
            if done:
                break
            state = next_state

    return w


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
            next_state, reward ,_= grid.move( action)
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
def get_all_states(shape):
    return [(i, j) for i in range(shape[0]) for j in range(shape[1])]


def main():
    parser = argparse.ArgumentParser(description="Simulate multi-armed bandit problem.")
    parser.add_argument('--episodes', type=int, default=10000, help='number of episodes')
    parser.add_argument('--alpha', type=float, default=0.1, help='epsilon') #0.1 0.2 0.3 0.5

    parser.add_argument('--gui', action='store_true' )
    parser.add_argument('--tune', action='store_true' )

    args = parser.parse_args()
    
    gui=args.gui
    alpha=args.alpha
    tune = args.tune
    n_episodes = args.episodes
    
    grid = Grid()
    GMC_w = gradient_MC(grid,n_episodes,alpha)
    print(GMC_w)
    print("-"*15)
    value_GMC = compute_grid_value_function(grid,GMC_w)
    print(value_GMC)

    grid = Grid()
    SGTD_w = semi_gradient_TD(grid,n_episodes,alpha)
    print(SGTD_w)
    print("-"*15)
    value_SGTD= compute_grid_value_function(grid,SGTD_w)
    print(value_SGTD)

    true_values = value_function_with_solving_bellman_eq(grid)
    print(true_values.shape)
    print(value_GMC.shape)
    plot_values([value_GMC,value_SGTD,true_values.reshape((7,7))],["values of Gradient Monte Carlo", " values of Semi GRadient TD","Values of Bellman Equation"])
    plot_values_per_state([value_GMC,value_SGTD,true_values],["Gradient Monte Carlo", " Semi GRadient TD","True Values"])
if __name__=="__main__":
    main()