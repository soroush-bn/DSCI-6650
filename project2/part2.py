from modified_grid import ModifiedGrid
import numpy as np
import random
from utils import *
import argparse

def MC_with_exploring_start(grid:ModifiedGrid, discount_factor=0.95, num_episodes=10000,playing=False):
    
    policy_prob = [[{"left": 0.25, "right": 0.25, "up": 0.25, "down": 0.25} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    policy = [["right" for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]

    Q = [[{"left": 0, "right": 0, "up": 0, "down": 0} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    Returns = [[{"left": [], "right": [], "up": [], "down": []} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    rewards = []
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
        if playing : rewards.append(play(policy))
    return policy,Q,rewards
                


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
        


def MC_epsilon_soft(grid: ModifiedGrid, discount_factor=0.95, epsilon=0.1, num_episodes=10000,playing=False):
    policy_prob = [[{"left": 0.25, "right": 0.25, "up": 0.25, "down": 0.25} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    policy = [["right" for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]

    Q = [[{"left": 0, "right": 0, "up": 0, "down": 0} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    Returns = [[{"left": [], "right": [], "up": [], "down": []} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    rewards = []
    for e in range(num_episodes):
        if e% 1000 ==0 :
            print("elapsed: %"+ str(e/num_episodes *100))
        history = generate_episode_epsilon_soft(grid, policy_prob, epsilon)
        G = 0
        visited_state_actions = set()
        if playing : rewards.append(play(policy))
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

    return policy, Q,rewards

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


def MC_off_policy(grid: ModifiedGrid, discount_factor=0.95, num_episodes=10000,playing=False):
    policy_prob = [[{"left": 0.25, "right": 0.25, "up": 0.25, "down": 0.25} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    policy = [["right" for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]

    behavior_policy_prob = [[{"left": 0.25, "right": 0.25, "up": 0.25, "down": 0.25} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]


    C = [[{"left": 0, "right": 0, "up": 0, "down": 0} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]

    Q = [[{"left": 0, "right": 0, "up": 0, "down": 0} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    Returns = [[{"left": [], "right": [], "up": [], "down": []} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            policy[i][j]=max(Q[i][j], key=Q[i][j].get)
    rewards = []
    for e in range(num_episodes): #can be replaced with while
        # b= soft max policy
        if e% 1000 ==0 :
            print("elapsed: %"+ str(e/num_episodes *100))
        history = generate_episode_behavior_policy(grid, policy_prob)
        G = 0
        W=1 
        visited_state_actions = set()
        if playing : rewards.append(play(policy))
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

    return policy, Q,rewards





def main():
    parser = argparse.ArgumentParser(description="Simulate multi-armed bandit problem.")
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--episodes', type=int, default=10000, help='number of episodes')
    parser.add_argument('--epsilon', type=float, default=0.01, help='epsilon')

    parser.add_argument('--gui', action='store_true' )
    parser.add_argument('--tune', action='store_true' )
    parser.add_argument('-q1', '--question1', action='store_true')
    parser.add_argument('-q2', '--question2', action='store_true')
    parser.add_argument('-q3', '--question3', action='store_true')

    args = parser.parse_args()
    
    gamma = args.gamma
    gui=args.gui
    q1 =args.question1
    q2= args.question2
    q3 = args.question3 
    tune = args.tune
    epsilon = args.epsilon
    n_episodes = args.episodes

    if q1:
        grid =ModifiedGrid()
    
        optimal_policy_MC_ES ,Q_MC_ES,rewards_MCWES = MC_with_exploring_start(grid= grid,discount_factor= gamma,num_episodes=n_episodes)

        pprint(optimal_policy_MC_ES)
        pprint(Q_MC_ES )

        print("*"*10)

        grid =ModifiedGrid()
    
        optimal_policy_MC_epsilon_soft ,Q_MC_epsilon_soft,rewards_MCES = MC_epsilon_soft(grid=grid,discount_factor=gamma,epsilon=epsilon,num_episodes=n_episodes)

        pprint(optimal_policy_MC_epsilon_soft)
        pprint(Q_MC_epsilon_soft )

        if gui:
            plot_policies([optimal_policy_MC_ES,optimal_policy_MC_epsilon_soft],[MC_with_exploring_start.__name__,MC_epsilon_soft.__name__],grid)

        if tune : 
            print("start to tune params")
            policies_MC_ES = [] 
            policies_MC_epsilon_soft = []
            discount_factors = [0.95,0.75,0.50,0]
            avg_rewards_over_dfs_MC_ES = []
            avg_rewards_over_dfs_MC_WES = []
            for df in discount_factors:
                grid = ModifiedGrid()
                optimal_policy_MC_ES ,Q_MC_ES,rewards_MCWES = MC_with_exploring_start(grid,df,num_episodes=1000)
                grid = ModifiedGrid()
                optimal_policy_MC_epsilon_soft ,Q_MC_epsilon_soft,rewards_MCES = MC_epsilon_soft(grid,df,num_episodes=1000)
                policies_MC_ES.append(optimal_policy_MC_ES)
                policies_MC_epsilon_soft.append(optimal_policy_MC_epsilon_soft)
                avg_rewards_over_dfs_MC_ES.append(play(optimal_policy_MC_ES))
                avg_rewards_over_dfs_MC_WES.append(play(optimal_policy_MC_epsilon_soft))
            print(avg_rewards_over_dfs_MC_ES)
            print(avg_rewards_over_dfs_MC_WES)
            plot_policies(policies_MC_ES,["gamma=0.95","gamma=0.75","gamma=0.50","gamma=0"],grid=grid)
            plot_policies(policies_MC_epsilon_soft,["gamma=0.95","gamma=0.75","gamma=0.50","gamma=0"],grid=grid)
            epsilons = [0.01 , 0.1 ,0.2 ,0.3 ]
            policies_of_different_epsilons = []
            for e in epsilons : 
                grid = ModifiedGrid()
                optimal_policy_MC_epsilon_soft ,Q_MC_epsilon_soft,rewards_MCES = MC_epsilon_soft(grid,discount_factor= df,epsilon=e  ,num_episodes=1000)

                policies_of_different_epsilons.append(optimal_policy_MC_epsilon_soft)
            plot_policies(policies_of_different_epsilons,["epsilon=0.01" , "epsilon=0.1" ,"epsilon=0.2" ,"epsilon=0.3" ],grid)


            # plot_values(v_policy_evals,["gamma=0.95","gamma=0.70","gamma=0.50","gamma=0"])


    if q2 : 
        grid= ModifiedGrid()
        optimal_policy_MC_off ,Q_MC_off,rewards_MCOP = MC_off_policy(grid,gamma,n_episodes)

        pprint(optimal_policy_MC_off)
        pprint(Q_MC_off )

        print("*"*10)
        if gui:
            plot_policies([optimal_policy_MC_off],[MC_off_policy.__name__],grid)

        if tune: 
            pass
            

    if q3:
        grid= ModifiedGrid()
        grid.permute_enable =True
        v,policy,_  = policy_iteration(grid)
        pprint(policy)
        if gui:
            plot_policies([policy],["policy obtained by policy iteration on permutable grid"],grid=grid)





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
                    next_state, reward,_ = grid.move(action) # we dont need P (transitions probabilities are all 1 execpt from special cells that you know for sure you are on that state)???
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
    



        

def play(policy,n =100):

    rewards = []
    step_threshhold =100
    for i in range(n):
        grid= ModifiedGrid()
        terminal= False
        reward = 0 
        j=0 
        while  j<step_threshhold : 
            j+=1
            state = grid.current_state
            action = policy[state[0]][state[1]]
            next_state, r, terminal = grid.move(action)
            reward += r
            if  terminal: break
        rewards.append(reward)

    return sum(rewards)/n





if __name__=="__main__":
    main()


