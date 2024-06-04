import numpy as np
import matplotlib.pyplot as plt 
import argparse
import random
SAVE_DIR = ".\\results\\"
np.random.seed(42)
class Bandit:
    def __init__(self, k=10):
        self.k = k
        self.action_values = np.random.normal(0, 1, self.k) #q_true
        self.action_counts = np.zeros(k)
        self.q_estimates = np.zeros(k)
        self.preferences = np.zeros(k)
        self.alpha = 0.1  # Learning rate for gradient bandit
        self.epsilon = 0.3

    

    def select_action(self,type="greedy"):
        if type=="greedy":
            return np.argmax(self.q_estimates)  # Choose action with highest estimated value
        elif type=="epsilon_greedy":
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.k)
            else:
                return np.argmax(self.q_estimates)
        elif type == "gradient":
            exp_preferences = np.exp(self.preferences)
            action_probabilities = exp_preferences / np.sum(exp_preferences)
            return np.random.choice(self.k, p=action_probabilities)
        else:
            raise TypeError(f"No such type: {type}, please choose greedy, epsilon_greedy, or gradient")

    def update_estimate(self,action,reward):
        self.action_counts[action] +=1 
        self.q_estimates[action] += (1.0 / self.action_counts[action]) * (reward - self.q_estimates[action])

    def update_preferences(self, action, reward, avg_reward):
        one_hot = np.zeros(self.k)
        one_hot[action] = 1
        self.preferences += self.alpha * (reward - avg_reward) * (one_hot - np.exp(self.preferences) / np.sum(np.exp(self.preferences)))

    
    def step(self,action):
        reward = self.get_reward(action)
        return reward
    
    def run(self,steps=1000,init_q_estimate_type= "zero",action_selection_type= "greedy",epsilon=0.1,alpha=0.3,decreasing= False,non_stationary=None,gradual_type="drift"):
        # print(""*10)
        # print(self.action_values)

        self.epsilon = epsilon
        rewards = []
        optimal_actions = np.zeros(steps)
        optimal_action_count = 0
        self.set_q_estimate(init_q_estimate_type)
        self.alpha= alpha
        avg_reward = 0
        optimal_action = np.argmax(self.action_values)
        decay = self.epsilon /100

        for i in range(steps):
            if steps%100 == 0  and decreasing==True:
                self.epsilon -= decay
                # self.plot_distributions() #verify reward changes
            if non_stationary=="abrupt":
                self.abrupt_change()
            elif non_stationary=="gradual":
                self.gradual_change(type=gradual_type)
            action = self.select_action(type=action_selection_type)
            # print(f"action {str(action)} selected")
            reward = self.step(action) #todo
            # print(f"reward {str(reward)} got")
            if action == optimal_action:
                optimal_actions[i]=1
            ##not good design 
            if action_selection_type == "gradient":
                avg_reward += (reward - avg_reward) / (i + 1)
                self.update_preferences(action, reward, avg_reward)
            else:
                self.update_estimate(action, reward)
            rewards.append(reward)
            

        # optimal_action_percentage = optimal_action_count / steps
        # self.plot_distributions()
        return rewards,optimal_actions
            

    
    def get_reward(self, action):
        return np.random.normal(self.action_values[action], 1)

    def set_q_estimate(self,type="zero"): # type can be zero, optimistic and random
        if type=="zero":
            self.q_estimates = np.zeros(shape=self.q_estimates.shape)
        elif type=="random" : 
            self.q_estimates = np.random.normal(0, 1, self.k)
        elif type=="optimistic":
            max_q = self.action_values.max()
            self.q_estimates = np.ones(shape=self.q_estimates.shape)*max_q +1
        else:
            raise NameError(f"type: {type} not found")
    

    def plot_distributions(self):
        rewards_data = [np.random.normal(self.action_values[i], 1, 10000) for i in range(self.k)]
        plt.figure(figsize=(10, 6))
        plt.violinplot(rewards_data, showmeans=False, showmedians=True)
        plt.title('Reward Distributions for 10-arm Bandit')
        plt.xlabel('Arm')
        plt.ylabel('Reward Distribution')
        plt.xticks(np.arange(1, self.k + 1), [f'Arm {i+1}' for i in range(self.k)])
        plt.show()

    def pilot_run(self):
        alphas = [0.1,0.3,0.5,0.7,1]
        epsilons = [0.01,0.1,0.2,0.3]
        num_problems = 1
        steps = 1000
        num_problems =10000
        avg_rewards_epsilon_greedy = np.zeros(steps)
        plt.figure(figsize=(10, 6))
        average_over_alphas = np.zeros(len(alphas))
        for i in range(len(epsilons)):
            # self.plot_distributions()
            for _ in range(num_problems):
                rewards, _ = self.run(steps=steps, init_q_estimate_type="zero", action_selection_type="epsilon_greedy", epsilon=epsilons[i])
                
                avg_rewards_epsilon_greedy += rewards
            plt.plot(avg_rewards_epsilon_greedy/num_problems,label=epsilons[i])

        

        plt.xlabel('Steps')
        plt.ylabel('Average Reward')
        plt.title('Average Rewards of different epsilons')
        plt.legend()
        plt.show()


    def gradual_change(self,type = "drift"): #types : drift, revert
        
        kapa = 0.5
        if type=="drift":
            curr_epsilon = np.random.normal(0, 0.001)
            self.action_values += curr_epsilon 
        elif type=="revert":
            curr_epsilon = np.random.normal(0, 0.01)
            self.action_values = (self.action_values* kapa) +  curr_epsilon
        else: 
            raise TypeError(f"No type: {type} found, please use drift or revert")


    def abrupt_change(self):
        permute_chance = 0.005
        for i in range(len(self.action_values)): #todo should we swap all indicies ?
            p = np.random.rand()
            if p<= permute_chance:
                i, j = np.random.choice(range(self.k), size=2, replace=False)
                self.action_values[i],self.action_values[j]= self.action_values[j],self.action_values[i]

        
