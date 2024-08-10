import argparse
from grid import Grid
import numpy as np
import random
from utils import *
# np.random.seed(42)


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
    dict1=  Q1[state[0]][state[1]]
    dict2= Q2[state[0]][state[1]]
    Q = {key: dict1[key] + dict2[key] for key in dict1}

    greedy_action = max(Q,key=Q1[state[0]][state[1]].get )
    greedy_action = action_map[greedy_action]
    action_probs[greedy_action] += (1.0 - epsilon)
    return action_probs

#todo Sarsa
def sarsa(grid: Grid,n_episodes = 10000, alpha=0.1,epsilon=0.1,discount=0.95,eps_decay=0.00005,tune=False):
    
    grid.reset()
    print("starting SARSA \n")
    print("from state: " + str(grid.current_state))
    steps=[]
    sum_of_rewards = []
    best_policy = [["left","left","left","right","right"],["up","up","up","up","up"],["up","up","up","up","up"],["right","right","up","left","left"],["right","right","up","left","left"]]
    Q = [[{"left": 20, "right": 20, "up": 20, "down": 20} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]

    best_policy = [["left","left","left","right","right"],["up","up","up","up","up"],["up","up","up","up","up"],["right","right","up","left","left"],["right","right","up","left","left"]]
    
    for i in range(n_episodes):
        grid.current_state= grid.blue_pos
        
        if i>500 and tune:
            p= get_policy(Q)
            score = similarity(grid,p,best_policy)
            if score>14:
                return Q,steps,sum_of_rewards
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

            S_prime,R,terminal = grid.move(A)
            if step_counter>10000:
                terminal=True
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
def Qlearning(grid: Grid,n_episodes = 5000, alpha=0.1,epsilon=0.05,discount=0.95,eps_decay=0.00005,tune=False):
    grid.reset()

    # policy_prob = [[{"left": 0.25, "right": 0.25, "up": 0.25, "down": 0.25} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    # policy = [["right" for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    best_policy = [["left","left","left","right","right"],["up","up","up","up","up"],["up","up","up","up","up"],["right","right","up","left","left"],["right","right","up","left","left"]]

    Q = [[{"left": 20, "right": 20, "up": 20, "down": 20} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    # Returns = [[{"left": [], "right": [], "up": [], "down": []} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    # rewards = []
    steps=[]
    sum_of_rewards = []

    for i in range(n_episodes):
        grid.current_state= grid.blue_pos
        if i>500 and tune:
            p= get_policy(Q)
            score = similarity(grid,p,best_policy)
            if score>14:
                return Q,steps,sum_of_rewards
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
            if step_counter>10000:
                terminal=True
            reward+=R
            Q[S[0]][S[1]][A] = Q[S[0]][S[1]][A] + alpha*(R + discount* max(Q[S_prime[0]][S_prime[1]].values()) - Q[S[0]][S[1]][A])
            S = S_prime
            # A = A_prime
        steps.append(step_counter)
        sum_of_rewards.append(reward)
    return Q,steps,sum_of_rewards

# todo expected sarsa
def expected_sarsa(grid: Grid,n_episodes = 1000, alpha=0.1,epsilon=0.05,discount=0.95,eps_decay=0.00005):
   
    Q = [[{"left": 20, "right": 20, "up": 20, "down": 20} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
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
        action_map = {'left': 0, 'right':1,'up':2,'down':3}

        while not terminal: #or replace with steps ?
            step_counter+=1

            S_prime,R,terminal = grid.move(A)
            if step_counter>10000:
                terminal=True
            action_probs = epsilon_greedy_policy(S_prime,Q,epsilon)
            A_prime = random.choices(grid.action_set,action_probs,k=1)[0]
            expected_target = 0 
            for a in grid.action_set:
                expected_target+= action_probs[action_map[a]] * Q[S_prime[0]][S_prime[1]][a]
            Q[S[0]][S[1]][A] = Q[S[0]][S[1]][A] + alpha*(R + discount* expected_target  - Q[S[0]][S[1]][A])
            S = S_prime
            A = A_prime
            reward+=R
        steps.append(step_counter)
        sum_of_rewards.append(reward)
    return Q,steps,sum_of_rewards

# todo d learning
def Doublelearning(grid: Grid,n_episodes = 1000, alpha=0.1,epsilon=0.05,discount=0.95,eps_decay=0.00005):

    Q1 = [[{"left": 20, "right": 20, "up": 20, "down": 20} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
    Q2 = [[{"left": 20, "right": 20, "up": 20, "down": 20} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]
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
            if step_counter>10000:
                terminal=True
            reward+=R
            if np.random.binomial(1, 0.5, 1)==0:
                
                Q1[S[0]][S[1]][A] = Q1[S[0]][S[1]][A] + alpha*(R + discount* max(Q2[S_prime[0]][S_prime[1]].values()) - Q1[S[0]][S[1]][A])
            else:
                Q2[S[0]][S[1]][A] = Q2[S[0]][S[1]][A] + alpha*(R + discount* max(Q1[S_prime[0]][S_prime[1]].values()) - Q2[S[0]][S[1]][A])

            S = S_prime
            # A = A_prime
        steps.append(step_counter)
        sum_of_rewards.append(reward)
    Q = [
        [
            {key: (Q1[i][j][key] + Q2[i][j][key]) / 2 for key in Q1[i][j]}
            for j in range(len(Q1[i]))
        ]
        for i in range(len(Q1))
    ]
    return Q,steps,sum_of_rewards

def dyna_q(grid, n_episodes=1000, alpha=0.1, gamma=0.95, epsilon=0.1, n_planning_steps=10):
    Q = [[{"left": 20, "right": 20, "up": 20, "down": 20} for _ in range(grid.shape[1])] for _ in range(grid.shape[0])]

    Model = [[{action: (0, (i, j)) for action in grid.action_set} for j in range(grid.shape[1])] for i in range(grid.shape[0])]

    for i in range(n_episodes):
        if i% 100 ==0 :
            print("elapsed: %"+ str(i/n_episodes *100))  
        S = grid.reset()
        terminal = False
        while not  terminal:
            action_probs = epsilon_greedy_policy(S,Q,epsilon)
            A = random.choices(grid.action_set,action_probs,k=1)[0]
            S_prime, R,terminal = grid.move(A)

            Q[S[0]][S[1]][A] += alpha * (R + gamma * max(Q[S_prime[0]][S_prime[1]].values()) - Q[S[0]][S[1]][A])
            Model[S[0]][S[1]][A] = (R, S_prime)

            for _ in range(n_planning_steps):
                S_sim = (random.randint(0, grid.shape[0] - 1), random.randint(0, grid.shape[1] - 1))
                A_sim = random.choice(grid.action_set)
                R_sim, S_prime_sim = Model[S_sim[0]][S_sim[1]][A_sim]
                
                Q[S_sim[0]][S_sim[1]][A_sim] += alpha * (R_sim + gamma * max(Q[S_prime_sim[0]][S_prime_sim[1]].values()) - Q[S_sim[0]][S_sim[1]][A_sim])

            S = S_prime

    return Q

def similarity(grid,policy,true_policy):
    score = 0 
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if (i,j) in grid.red_states or (i,j) in grid.terminal_states:
                continue
            else:
                try:
                    if policy[i][j]==true_policy[i][j]:
                        score+=1
                except:
                    print(i)
                    print(j)
                    print(policy)
    return score


def main():
    parser = argparse.ArgumentParser(description="Simulate multi-armed bandit problem.")
    parser.add_argument('--episodes', type=int, default=2000, help='number of episodes')
    parser.add_argument('--epsilon', type=float, default=0.3, help='epsilon')
    parser.add_argument('--alpha', type=float, default=0.1, help='epsilon')

    parser.add_argument('--gui', action='store_true' )
    parser.add_argument('--tune', action='store_true' )

    args = parser.parse_args()
    
    gui=args.gui
    alpha=args.alpha
    tune = args.tune
    epsilon = args.epsilon
    n_episodes = args.episodes
    grid= Grid()
    true_values = value_function_with_solving_bellman_eq(grid).reshape(5,5)
    print(true_values)


    if tune:
        Q_sarsa_tune= [] 
        rs_tune = [] 
        Q_qlearning_tune= [] 
        rqq_tune=  [] 
        for alpha in [0.1,0.2,0.3,0.4,0.5]:
            Q_sarsa,steps_sarsa,rewards_sarsa= sarsa(grid,n_episodes,alpha,epsilon)
            Q_Qlearning,steps_qlearning,rewards_qlearning = Qlearning(grid,n_episodes,alpha,epsilon)
            Q_sarsa_tune.append(Q_sarsa)
            Q_qlearning_tune.append(Q_Qlearning)
            rs_tune.append(rewards_sarsa)
            rqq_tune.append(rewards_qlearning)

        plot_rewards([*rs_tune,*rqq_tune],["sarsa a=0.1","sarsa a=0.2","sarsa a=0.3","sarsa a=0.4","sarsa a=0.5","qlearning a=.01","qlearning a=.02","qlearning a=.03","qlearning a=.04","qlearning a=.05"])
        play(grid,Q_sarsa_tune,["sarsa a=0.1","sarsa a=0.2","sarsa a=0.3","sarsa a=0.4","sarsa a=0.5"])
        play(grid,Q_qlearning_tune,["qlearning a=.01","qlearning a=.02","qlearning a=.03","qlearning a=.04","qlearning a=.05"])


    ##Sarsa
    grid = Grid()
    Q_sarsa,steps_sarsa,rewards_sarsa= sarsa(grid,n_episodes,alpha,epsilon)
    print("finished Sarsa: final q: ")
    print(Q_sarsa)
    # plot_state_values(Q_sarsa)
    # plot_policy(grid,Q_sarsa)


    ## Q learning
    grid = Grid()
    Q_Qlearning,steps_qlearning,rewards_qlearning = Qlearning(grid,n_episodes,alpha,epsilon)
    print("finished Q learning: final Q: ")
    print(Q_Qlearning)
    # plot_state_values(Q_sarsa)
    # plot_policy(grid,Q_Qlearning)

    # Dyna Q
    # grid= Grid()
    # Q_DynaQ = dyna_q(grid,n_episodes,alpha=alpha,epsilon=epsilon,n_planning_steps=10)
    # print("finished dyna q")
    # print(Q_DynaQ)
    # play(grid,[Q_DynaQ],["dyna q"])
    # plot_policies_grid(grid,[Q_DynaQ],["dyna Q"])
    
        # Q learning
    grid = Grid()
    Q_ExpectedSarsa,steps_ExpectedSarsa,rewards_ExpectedSarsa = expected_sarsa(grid,n_episodes,alpha,epsilon)
    print("finished expected sarsa: final Q: ")
    print(Q_ExpectedSarsa)

    ## qqlearning
    grid = Grid()
    Q_DQlearning,steps_DQlearning,rewards_DQlearning = Doublelearning(grid,n_episodes,alpha,epsilon)
    print("finished expected sarsa: final Q: ")
    print(Q_DQlearning)

    play(grid,[Q_sarsa,Q_Qlearning,Q_ExpectedSarsa,Q_DQlearning],["Sarsa Policy","Q-learning Policy","Expected SARSA Policy","Double Q-learning Policy"])

    plot_policies_grid(grid,[Q_sarsa,Q_Qlearning,Q_ExpectedSarsa,Q_DQlearning],["Sarsa Policy","Q-learning Policy","Expected SARSA Policy","Double Q-learning Policy"])
    plot_time_steps([steps_sarsa,steps_qlearning,steps_ExpectedSarsa,steps_DQlearning],["steps Sarsa","steps Q-learning","steps Expected Sarsa","steps Double Q-learning"])
    plot_rewards([rewards_sarsa,rewards_qlearning,rewards_ExpectedSarsa,rewards_DQlearning],["rewards Sarsa","rewards Q-learning","rewards Expected Sarsa","rewards Double Q-learning"])



def play(grid,QS,titles):
    for Q,t in zip(QS,titles):
        terminal = False
        reward = 0
        steps = 0 
        pre_a=  None
        A= None
        state= grid.reset()
        trajectory = []
        while not terminal:
            steps+=1
            if steps>100:
                print("policy " + str(t) + "stuck in" + str(state))
                break
            pre_a=A
            A = max(Q[state[0]][state[1]],key=Q[state[0]][state[1]].get )
            trajectory.append(A)      
            S_prime , R , terminal =  grid.move(A)
            reward += R

            state=S_prime
        print("policy of " + str(t) +" finished in " + str(steps) + " steps with reward of: " + str(reward))
        print()



if __name__=="__main__":
    main()