from grid import Grid
import random
import numpy as np
def random_policy(actions):
    return random.choice(actions)



def main():
    grid =Grid()
    gamma = 0.95
    v = value_function_with_solving_bellman_eq(grid,gamma)
    arr = np.array(v)
    print("*" *12)
    print("Estimating V function with "  +str(value_function_with_solving_bellman_eq.__name__))
    print("max position = " + str(np.argmax(v)))
    print("values: \n")
    print(v)
    print("--"*12)

    # grid =Grid()
    # v = policy_improvement_with_value_iteration(grid,gamma)
    # arr = np.array(v)
    # print("*" *12)
    # print("Estimating V function with "  +str(policy_improvement_with_value_iteration.__name__))
    # print("max position = " + str(np.argmax(v)))
    # print("values: \n")
    # print(v)
    # print("--"*12)


    grid =Grid()
    v = value_function_with_iterative_policy_evaluation(grid,gamma)
    arr = np.array(v)
    print("*" *12)
    print("Estimating V function with "  +str(value_function_with_iterative_policy_evaluation.__name__))
    print("max position = " + str(np.argmax(v)))
    print("values: \n")
    print(v)
    print("--"*12)

def get_all_states(shape):
    return [(i, j) for i in range(shape[0]) for j in range(shape[1])]

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
            P[state_idx, next_state_idx] += 0.25  # Equal probability for all actions
            R[state_idx] += 0.25 * reward  # Average reward for all actions

    return P, R

def value_function_with_solving_bellman_eq(grid, discount_factor=0.95):
    P, R = create_transition_and_reward_matrices(grid)
    num_states = P.shape[0]
    I = np.eye(num_states)
    V = np.linalg.solve(I - discount_factor * P, R)
    return V

def arg_max_from_value_function(value_function,grid):
    def max_action(up,down,left,right):
        if up>= down and up >= left and up>=right : 
            return "up"
        elif down >=up and down>= right and down >= left : 
            return "down"
        elif right >= left and right >=down and right>= up:
            return "right"
        else:
            return "left"

    policy = [[0]*grid.shape[1] for _ in range(grid.shape[0])]


    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            up=down=right=left = -999

            if i-1 >=0:
                up = value_function[i-1][j] # up
            if i+1 < grid.shape[0]:
                down = value_function[i+1][j] # down
            if j-1>=0:
                left = value_function[i][j-1] # left
            if j+1 < grid.shape[1]: 
                right =  value_function[i][j+1] # right
            
            policy[i][j] =  max_action(up,down,left,right)

    return policy


def deterministic_policy_to_policy_probs(deterministic_policy,grid):

    policy_prob = [[0]*grid.shape[0]]*grid.shape[1]
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            p = deterministic_policy[i][j]
            policy_prob[i][j]= {"left":0,"right":0,"up":0,"down": 0}
            policy_prob[i][j][p] = 1
    return policy_prob

def policy_prob_to_deterministic_policy(policy_prob, grid):
    deterministic_policy = [[0]*grid.shape[1] for _ in range(grid.shape[0])]
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            deterministic_policy[i][j] = max(policy_prob[i][j], key=policy_prob[i][j].get)
    return deterministic_policy


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

def policy_iteration(grid,discount_factor = 0.95):
    # init 
    v = np.zeros(grid.shape)
    policy_prob = [[0]*grid.shape[0]]*grid.shape[1]
    policy = [["right"]*grid.shape[1] for _ in range(grid.shape[0])]
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            policy_prob[i][j]= {"left":0.25,"right":0.25,"up":0.25,"down": 0.25}# Equal probability for each action

    ### does argmax means the policy(s) should only return one action at a time or it can also return a probability set ??
    while True : 
        #policy evaluation 
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
                    return value_function,policy
    


            



def policy_improvement_with_value_iteration(grid, discount_factor=0.95, theta=1e-5):
    shape = grid.shape
    value_function = np.zeros(shape)
    while True:
        delta = 0 
        for i in range(shape[0]):
            for j in range(shape[1]):
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
        if delta < theta:
                break
            
    policy = arg_max_from_value_function(value_function,grid)

    return value_function,policy


                



if __name__=="__main__":
    main()