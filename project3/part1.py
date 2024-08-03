from grid import Grid
import numpy as np
import random
def epsilon_greedy_policy(state,Q,epsilon):
    action_probs = np.array([epsilon/4,epsilon/4,epsilon/4,epsilon/4]) # ?make sense?
    greedy_action= np.argmax(Q[state])
    action_probs[greedy_action] = 1 - epsilon
    return action_probs

#todo Sarsa
def sarsa(grid: Grid,n_episodes = 1000, alpha=0.1,epsilon=0.05,discount=0.95):
    # policy_prob = [[{"left": 0.25, "right": 0.25, "up": 0.25, "down": 0.25} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    # policy = [["right" for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]

    Q = [[{"left": 0, "right": 0, "up": 0, "down": 0} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    # Returns = [[{"left": [], "right": [], "up": [], "down": []} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    # rewards = []

    for i in n_episodes:
        terminal = False
        S = grid.current_state
        action_probs = epsilon_greedy_policy(S,Q,epsilon)
        A = random.choice(grid.action_set,p = action_probs)
        while not terminal: #or replace with steps ?
            S_prime,R,terminal = grid.move(A)
            action_probs = epsilon_greedy_policy(S_prime,Q,epsilon)
            A_prime = random.choice(grid.action_set,p = action_probs)
            Q[S][A] = Q[S][A] + alpha(R + discount* Q[S_prime][A_prime] - Q[S][A])
            S = S_prime
            A = A_prime
        
        return Q
    
    
             


    







# todo Q Learning

# todo expected sarsa

# todo d learning

def main():
    pass

if __name__=="__main__":
    main()