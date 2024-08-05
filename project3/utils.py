import numpy as np
from matplotlib import pyplot as plt 
import random
import seaborn as sns




def get_state_values(action_values):

    n_rows = len(action_values)
    n_cols = len(action_values[0])
    
    state_values = np.zeros((n_rows, n_cols))

    for row in range(n_rows):
        for col in range(n_cols):
            state = action_values[row][col]
            state_values[row, col] = max(state.values())
    
    return state_values



def get_policy(Q):
    n_rows = len(Q)
    n_cols = len(Q[0])
    policy_grid = np.empty((n_rows, n_cols), dtype=str)
    
    for row in range(n_rows):
        for col in range(n_cols):
            state = Q[row][col]
            best_action = max(state, key=state.get)
            policy_grid[row, col] = best_action
    policy_grid


def plot_policy(grid,action_values):

    n_rows = len(action_values)
    n_cols = len(action_values[0])
    actions = ['left', 'right', 'up', 'down']
    action_arrows = {"right": "→", "down": "↓", "up": "↑", "left": "←"}
    
    policy_grid = np.empty((n_rows, n_cols), dtype=str)
    
    for row in range(n_rows):
        for col in range(n_cols):
            state = action_values[row][col]
            best_action = max(state, key=state.get)
            policy_grid[row, col] = action_arrows[best_action]
    
    # Plot the grid with arrows indicating the actions
    fig, ax = plt.subplots(figsize=(n_cols, n_rows))
    ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(np.arange(n_cols))
    ax.set_yticklabels(np.arange(n_rows))
    
    for row in range(n_rows):
        for col in range(n_cols):
            ax.text(col, row, policy_grid[row, col], ha='center', va='center', fontsize=20)
            if (row,col) in grid.red_states:
                ax.add_patch(plt.Rectangle((col-0.5, row-0.5), 1, 1, edgecolor='black', facecolor="red"))
            elif (row,col) in grid.terminal_states:
                ax.add_patch(plt.Rectangle((col-0.5, row-0.5), 1, 1, edgecolor='black', facecolor="black"))
            elif (row,col) ==  grid.blue_pos:
                ax.add_patch(plt.Rectangle((col-0.5, row-0.5), 1, 1, edgecolor='black', facecolor="blue"))
                
    plt.gca().invert_yaxis()
    plt.show()

def plot_state_values(Q):
    state_values = get_state_values(Q)
    plt.figure(figsize=(10, 8))
    plt.imshow(state_values, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='State Value')
    plt.title('State Values Grid')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    
    for i in range(state_values.shape[0]):
        for j in range(state_values.shape[1]):
            plt.text(j, i, f'{state_values[i, j]:.2f}', ha='center', va='center', color='white')
    
    plt.gca().invert_yaxis()
    plt.show()





def plot_policies_grid(grid,q_values_list, titles_list):

    n_policies = len(q_values_list)
    n_cols = 2
    n_rows = (n_policies + 1) // 2  
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    axs = axs.flatten()  
    
    actions = ['left', 'right', 'up', 'down']
    action_arrows = {"right": "→", "down": "↓", "up": "↑", "left": "←"}
    
    for idx, (action_values, title) in enumerate(zip(q_values_list, titles_list)):
        ax = axs[idx]
        n_rows_grid = grid.shape[0]
        n_cols_grid = grid.shape[1]
        
        policy_grid = np.empty((n_rows_grid, n_cols_grid), dtype=str)
        
        for row in range(n_rows_grid):
            for col in range(n_cols_grid):
                state = action_values[row][col]
                best_action = max(state, key=state.get)
                policy_grid[row, col] = action_arrows[best_action]
        
        ax.set_xticks(np.arange(n_cols_grid + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_rows_grid + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
        ax.tick_params(which="minor", size=0)
        ax.set_xticks(np.arange(n_cols_grid))
        ax.set_yticks(np.arange(n_rows_grid))
        ax.set_xticklabels(np.arange(n_cols_grid))
        ax.set_yticklabels(np.arange(n_rows_grid))
        
        for row in range(n_rows_grid):
            for col in range(n_cols_grid):
                ax.text(col, row, policy_grid[row, col], ha='center', va='center', fontsize=20)
                if (row,col) in grid.red_states:
                    ax.add_patch(plt.Rectangle((col-0.5, row-0.5), 1, 1, edgecolor='black', facecolor="red"))
                elif (row,col) in grid.terminal_states:
                    ax.add_patch(plt.Rectangle((col-0.5, row-0.5), 1, 1, edgecolor='black', facecolor="black"))
                elif (row,col) ==  grid.blue_pos:
                    ax.add_patch(plt.Rectangle((col-0.5,row-0.5), 1, 1, edgecolor='black', facecolor="blue"))
                    
        
        ax.set_title(title)
        ax.invert_yaxis()
    
    for idx in range(n_policies, len(axs)):
        axs[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


    #todo sum of rewards / episodes 
def plot_rewards(rewards_list, titles_list):
    stepping_factor = 20

    plt.figure(figsize=(10, 6))
    for time_steps, title in zip(rewards_list, titles_list):
        # Average the rewards in chunks of 50
        reshaped = np.reshape(time_steps[:len(time_steps) - len(time_steps) % stepping_factor], (-1, stepping_factor))
        averaged_rewards = np.mean(reshaped, axis=1)
        
        episodes = list(range(1, len(averaged_rewards) + 1))
        plt.plot(episodes, averaged_rewards, label=title)
    
    plt.xlabel('Episode Number')
    plt.ylabel('Average Sum of Rewards')
    plt.title('Average Sum of Rewards per Episode')
    plt.legend()
    plt.grid(True)
    plt.show()

    #todo episodes / number of time steps 
    # 
def plot_time_steps(time_steps_list, titles_list):
    stepping_factor = 20

    plt.figure(figsize=(10, 6))
    for time_steps, title in zip(time_steps_list, titles_list):
        # Average the time steps in chunks of 50
        reshaped = np.reshape(time_steps[:len(time_steps) - len(time_steps) % stepping_factor], (-1, stepping_factor))
        averaged_time_steps = np.mean(reshaped, axis=1)
        
        episodes = list(range(1, len(averaged_time_steps) + 1))
        plt.plot(episodes, averaged_time_steps, label=title)
    
    plt.xlabel('Episode Number')
    plt.ylabel('Average Time Steps')
    plt.title('Average Time Steps per Episode')
    plt.legend()
    plt.grid(True)
    plt.show()
    # todo sum of rewards per episode for different alphas