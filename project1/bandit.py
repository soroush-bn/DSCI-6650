import numpy as np
import matplotlib.pyplot as plt 

SAVE_DIR = ".\results"

class Bandit:
    def __init__(self, k=10):
        self.k = k
        self.action_values = np.random.normal(0, 1, self.k) #q_true
        self.action_counts = np.zeros(k)
        self.q_estimates = np.zeros(k)
        self.preferences = np.zeros(k)
        self.alpha = 0.1  # Learning rate for gradient bandit
        self.epsilon = 0.3

    

    def select_action(self,type="greedy",):
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
    
    def run(self,steps=1000,init_action_value_type= "zero",action_selection_type= "greedy",epsilon=0.1,alpha=0.3):
        # print(""*10)
        # print(self.action_values)


        rewards = []
        optimal_action_count = 0
        # self.set_action_values(init_action_value_type)
        self.alpha= alpha
        avg_reward = 0
        optimal_action = np.argmax(self.action_values)
        # print(optimal_action)
        for i in range(steps):
            action = self.select_action(type=action_selection_type)
            # print(f"action {str(action)} selected")
            reward = self.step(action) #todo
            # print(f"reward {str(reward)} got")
            if action == optimal_action:
                optimal_action_count += 1
            ##not good design 
            if action_selection_type == "gradient":
                avg_reward += (reward - avg_reward) / (i + 1)
                self.update_preferences(action, reward, avg_reward)
            else:
                self.update_estimate(action, reward)
            rewards.append(reward)

        optimal_action_percentage = optimal_action_count / steps
        # self.plot_distributions()
        return rewards,optimal_action_percentage
            

    
    def get_reward(self, action):
        return np.random.normal(self.action_values[action], 1)

    def set_action_values(self,type="zero"): # type can be zero, optimistic and random
        if type=="default": return
        if type=="zero":
            self.action_values = np.zeros(shape=self.action_values.shape)
        elif type=="random" : 
            self.action_values = np.random.normal(0, 1, self.k)
        # elif type=="optimistic":
        #     self.action_values = np.ones(shape = self.action_values.shape)*2
    

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
        epsilons = [0.1,0.3,0.5,0.7,1]
        num_problems = 1
        steps = 1000
        num_problems =1000 
        avg_rewards_epsilon_greedy = np.zeros((len(epsilons),steps))
        plt.figure(figsize=(10, 6))
        average_over_alphas = np.zeros(len(alphas))
        for i in range(len(epsilons)):
            # self.plot_distributions()
            for _ in range(num_problems):
                rewards, _ = self.run(steps=steps, init_action_value_type="default", action_selection_type="greedy", epsilon=epsilons[i])
                print(len(rewards))
                avg_rewards_epsilon_greedy[i] += rewards
                plt.plot(avg_rewards_epsilon_greedy[i],label=epsilons[i])

        

        plt.xlabel('Steps')
        plt.ylabel('Average Reward')
        plt.title('Average Rewards of different epsilons')
        plt.legend()
        plt.show()

                

def main():
    num_problems = 1000
    steps = 1000

    avg_rewards_greedy = np.zeros(steps)
    avg_rewards_epsilon_greedy = np.zeros(steps)
    avg_rewards_optimistic = np.zeros(steps)
    avg_rewards_gradient = np.zeros(steps)

    optimal_action_percentage_greedy = np.zeros(steps)
    optimal_action_percentage_epsilon_greedy = np.zeros(steps)
    optimal_action_percentage_optimistic = np.zeros(steps)
    optimal_action_percentage_gradient = np.zeros(steps)

    epsilon = 0.1  # Example epsilon value; you can run pilot tests to determine the best value
    alpha = 0.1    # Learning rate for the gradient bandit algorithm

    for i in range(num_problems):
        bandit = Bandit()
        rewards, optimal_action_percentage = bandit.run(steps=steps, init_action_value_type="zero", action_selection_type="greedy")
        avg_rewards_greedy += rewards
        optimal_action_percentage_greedy += optimal_action_percentage

        rewards, optimal_action_percentage = bandit.run(steps=steps, init_action_value_type="zero", action_selection_type="epsilon_greedy", epsilon=epsilon)
        avg_rewards_epsilon_greedy += rewards
        optimal_action_percentage_epsilon_greedy += optimal_action_percentage

        rewards, optimal_action_percentage = bandit.run(steps=steps, init_action_value_type="optimistic", action_selection_type="greedy")
        avg_rewards_optimistic += rewards
        optimal_action_percentage_optimistic += optimal_action_percentage

        rewards, optimal_action_percentage = bandit.run(steps=steps, init_action_value_type="zero", action_selection_type="gradient", epsilon=epsilon)
        avg_rewards_gradient += rewards
        optimal_action_percentage_gradient += optimal_action_percentage

    avg_rewards_greedy /= num_problems
    avg_rewards_epsilon_greedy /= num_problems
    avg_rewards_optimistic /= num_problems
    avg_rewards_gradient /= num_problems

    optimal_action_percentage_greedy /= num_problems
    optimal_action_percentage_epsilon_greedy /= num_problems
    optimal_action_percentage_optimistic /= num_problems
    optimal_action_percentage_gradient /= num_problems

    plt.figure(figsize=(10, 6))
    plt.plot(avg_rewards_greedy, label="Greedy")
    plt.plot(avg_rewards_epsilon_greedy, label="Epsilon-Greedy")
    plt.plot(avg_rewards_optimistic, label="Optimistic")
    plt.plot(avg_rewards_gradient, label="Gradient Bandit")
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs. Steps for Different Bandit Strategies')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(optimal_action_percentage_greedy, label="Greedy")
    plt.plot(optimal_action_percentage_epsilon_greedy, label="Epsilon-Greedy")
    plt.plot(optimal_action_percentage_optimistic, label="Optimistic")
    plt.plot(optimal_action_percentage_gradient, label="Gradient Bandit")
    plt.xlabel('Steps')
    plt.ylabel('Optimal Action Percentage')
    plt.title('Optimal Action Percentage vs. Steps for Different Bandit Strategies')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Bandit().pilot_run()
    main()
