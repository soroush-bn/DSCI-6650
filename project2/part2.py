from modified_grid import ModifiedGrid
import numpy as np
import random
from utils import *
import argparse

def MC_with_exploring_start(grid:ModifiedGrid, discount_factor=0.95, theta=1e-3, num_episodes=1000):
    
    policy_prob = [[{"left": 0.25, "right": 0.25, "up": 0.25, "down": 0.25} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    policy = [["right" for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]

    Q = [[{"left": 0, "right": 0, "up": 0, "down": 0} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    Returns = [[{"left": [], "right": [], "up": [], "down": []} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]

    for e in range(num_episodes): 
        if e% 1000 ==0 :
            print("elapsed: %"+ str(e/num_episodes *100))       
        i0 = random.choice(list(range(grid.shape[0])))
        j0 = random.choice(list(range(grid.shape[1])))
        A0 = random.choice(grid.action_set)
        history = generate_episode(grid,(i0,j0),A0,policy_prob)
        G=0
        # for (s,a,r) in history:
        visited_state_actions = []
        for t in range(len(history)-1, -1, -1):
            s,a,r =history[t]
            G = discount_factor*G +r
            if ( s, a) not in visited_state_actions:
                visited_state_actions.append( (s,a))
                Returns[s[0]][s[1]][a].append(G)
                Q[s[0]][s[1]][a] = np.mean(Returns[s[0]][s[1]][a])
                policy[s[0]][s[1]] = max(Q[s[0]][s[1]], key=Q[s[0]][s[1]].get)

    return policy,Q
                


def generate_episode(grid: ModifiedGrid, S0, A0, policy_probs):
    # Generate an episode: (S0, A0, R1, ..., St-1, At-1, Rt)
    grid.current_state = S0
    history = []
    terminal = False
    action = A0
    while not terminal:
        pre_s = grid.current_state
        pre_a = action
        next_state, reward, terminal = grid.move(action)
        history.append((pre_s, pre_a, reward))
        action = choose_action(policy_probs, next_state)
    
    return history
        


def MC_epsilon_soft(grid: ModifiedGrid, discount_factor=0.95, epsilon=0.1, num_episodes=1000):
    policy_prob = [[{"left": 0.25, "right": 0.25, "up": 0.25, "down": 0.25} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    policy = [["right" for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]

    Q = [[{"left": 0, "right": 0, "up": 0, "down": 0} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    Returns = [[{"left": [], "right": [], "up": [], "down": []} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]

    for e in range(num_episodes):
        if e% 1000 ==0 :
            print("elapsed: %"+ str(e/num_episodes *100))
        history = generate_episode_epsilon_soft(grid, policy_prob, epsilon)
        G = 0
        visited_state_actions = set()

        for t in range(len(history)-1, -1, -1):
            s, a, r = history[t]
            G = discount_factor * G + r
            if (s, a) not in visited_state_actions:
                visited_state_actions.add((s, a))
                Returns[s[0]][s[1]][a].append(G)
                Q[s[0]][s[1]][a] = np.mean(Returns[s[0]][s[1]][a])
                policy[s[0]][s[1]] = max(Q[s[0]][s[1]], key=Q[s[0]][s[1]].get)

                for action in policy_prob[s[0]][s[1]]:
                    if action == policy[s[0]][s[1]]:
                        policy_prob[s[0]][s[1]][action] = 1 - epsilon + (epsilon / len(policy_prob[s[0]][s[1]]))
                    else:
                        policy_prob[s[0]][s[1]][action] = epsilon / len(policy_prob[s[0]][s[1]])

    return policy, Q

def epsilon_greedy_action(policy_probs, state, epsilon):
    i, j = state
    actions = list(policy_probs[i][j].keys())
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    else:
        return max(policy_probs[i][j], key=policy_probs[i][j].get)



def generate_episode_epsilon_soft(grid: ModifiedGrid, policy_probs, epsilon):
    # Generate an episode with epsilon-soft policy
    state = grid.current_state
    history = []
    terminal = False
    while not terminal:
        action = epsilon_greedy_action(policy_probs, state, epsilon)
        next_state, reward, terminal = grid.move(action)
        history.append((state, action, reward))
        state = next_state
    return history


def generate_episode_behavior_policy(grid: ModifiedGrid, behavior_policy_probs):
    state = grid.current_state
    history = []
    terminal = False
    while not terminal:
        action = choose_action(behavior_policy_probs, state)
        next_state, reward, terminal= grid.move(action)
        history.append((state, action, reward))
        state = next_state
    return history


def MC_off_policy(grid: ModifiedGrid, discount_factor=0.95, epsilon=0.1, num_episodes=1000):
    policy_prob = [[{"left": 0.25, "right": 0.25, "up": 0.25, "down": 0.25} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    policy = [["right" for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]

    behavior_policy_prob = [[{"left": 0.25, "right": 0.25, "up": 0.25, "down": 0.25} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]


    C = [[{"left": 0, "right": 0, "up": 0, "down": 0} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]

    Q = [[{"left": 0, "right": 0, "up": 0, "down": 0} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    Returns = [[{"left": [], "right": [], "up": [], "down": []} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            policy[i][j]=max(Q[i][j], key=Q[i][j].get)

    for e in range(num_episodes): #can be replaced with while
        # b= soft max policy
        if e% 1000 ==0 :
            print("elapsed: %"+ str(e/num_episodes *100))
        history = generate_episode_epsilon_soft(grid, policy_prob, epsilon)
        G = 0
        W=1 
        visited_state_actions = set()

        for t in range(len(history)-1, -1, -1):
            s, a, r = history[t]
            G = discount_factor * G + r
            
            # if (s, a) not in visited_state_actions:
                # visited_state_actions.add((s, a))
            # Returns[s[0]][s[1]][a].append(G)
            C[s[0]][s[1]][a] = C[s[0]][s[1]][a] +W
            Q[s[0]][s[1]][a] +=(W/C[s[0]][s[1]][a])*(G-Q[s[0]][s[1]][a] ) 
            policy[s[0]][s[1]] = max(Q[s[0]][s[1]], key=Q[s[0]][s[1]].get)
            if a != policy[s[0]][s[1]] : 
                break
            W = W * (1.0 / behavior_policy_prob[i][j][a])

    return policy, Q





def main():
    parser = argparse.ArgumentParser(description="Simulate multi-armed bandit problem.")
    parser.add_argument('--theta', type=float, default=1e-3, help='theta threshold')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--episodes', type=int, default=10000, help='number of episodes')

    parser.add_argument('--gui', action='store_true' )
    parser.add_argument('-q1', '--question1', action='store_true')
    parser.add_argument('-q2', '--question2', action='store_true')
    args = parser.parse_args()
    
    gamma = args.gamma
    theta=args.theta
    gui=args.gui
    q1 =args.question1
    q2= args.question2
    n_episodes = args.episodes

    if q1:
        grid =ModifiedGrid()
    
        optimal_policy_MC_ES ,Q_MC_ES = MC_with_exploring_start(grid,gamma,theta,n_episodes)

        pprint(optimal_policy_MC_ES)
        pprint(Q_MC_ES )

        print("*"*10)

        grid =ModifiedGrid()
    
        optimal_policy_MC_epsilon_soft ,Q_MC_epsilon_soft = MC_epsilon_soft(grid,gamma,theta,n_episodes)

        pprint(optimal_policy_MC_epsilon_soft)
        pprint(Q_MC_epsilon_soft )




        if gui:
            plot_policies([optimal_policy_MC_ES,optimal_policy_MC_epsilon_soft],[MC_with_exploring_start.__name__,MC_epsilon_soft.__name__])



if __name__=="__main__":
    main()


