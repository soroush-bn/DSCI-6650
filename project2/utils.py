import numpy as np
from matplotlib import pyplot as plt 
import random
import seaborn as sns

def print_details(value_function,title,grid):
        
        print("*" *12)
        print("Estimating V function with "  +str(title))
        print("max position = " + str(np.argmax(value_function)))
        print("values: \n")
        pprint(value_function.reshape(grid.shape).tolist())
        print("--"*12)

def plot_values(values,titles ):
    fig, axs = plt.subplots(1, len(values), figsize=(16, 6))
    for i in range(len(values)):
        sns.heatmap(values[i], annot=True, fmt=".2f", cmap='viridis', cbar=True, ax=axs[i])
        axs[i].set_title(titles[i])
        axs[i].set_xlabel('Column')
        axs[i].set_ylabel('Row')
    plt.show()

def plot_policies(policy_grids, titles=None,grid=None):
    num_policies = len(policy_grids)
    fig, axs = plt.subplots(1, num_policies, figsize=(8 * num_policies, 8))
    if len(policy_grids)==1 : axs = [axs]
    if titles is None:
        titles = [f'Policy {i+1}' for i in range(num_policies)]
    
    orientation_mapper = {"right": "→", "down": "↓", "up": "↑", "left": "←"}

    def plot_policy(ax, policy_grid, title,colors):
        n, m = len(policy_grid), len(policy_grid[0])
        colors = colors

        for i in range(n):
            for j in range(m):
                color = colors.get((i, j), "white")
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, edgecolor='black', facecolor=color))
                ax.text(j, i, orientation_mapper[policy_grid[i][j]], ha='center', va='center', fontsize=12)
        ax.set_xlim(-0.5, m-0.5)
        ax.set_ylim(-0.5, n-0.5)
        ax.set_xticks(np.arange(-0.5, m, 1))
        ax.set_yticks(np.arange(-0.5, n, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True)
        ax.set_title(title)
        ax.invert_yaxis()
    
    for i in range(num_policies):
        plot_policy(axs[i], policy_grids[i], titles[i],grid.color_map)
    
    plt.show()


def random_policy(actions):
    return random.choice(actions)


def pprint(policy):
    for i in range(5):
        print(policy[i])
        print()


def get_all_states(shape):
    return [(i, j) for i in range(shape[0]) for j in range(shape[1])]




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


def choose_action(policy_prob, state):
    i, j = state
    actions = list(policy_prob[i][j].keys())
    probabilities = list(policy_prob[i][j].values())
    chosen_action = random.choices(actions, probabilities)[0]
    return chosen_action

