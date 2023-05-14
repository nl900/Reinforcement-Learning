import gym
import numpy as np
import random

"""
Q-learning learns to behave optimally by repeatedly interacting with the env.
It learns the value of an action in a particular state called Q-values or Action-values
Q(s,a) which is the immediate reward/penalty and estimate of the value of the next state.
It does not require model of the env (model free) and it learns by taking random actions which 
is not the optimal policy being evaluated and improved (off-policy)
"""

def train(env, qtable):
    def selectAction(state):
        if random.uniform(0,1) < epsilon:
            return env.action_space.sample() # explore, randomly sample from available actions
        else:
            return np.argmax(qtable[state])  # exploit, select action max future expecture reward
    
    # hyperparameters
    learning_rate = 0.001  
    discount_rate = 0.8  # how much to weigh future rewards
  
    num_episodes = 10000    # training variables
    
    # exploration-exploitation tradeoff
    # during exploitation, the agent select the action with the highest Q-value
    # over time, the agen will explore less and start exploiting
    epsilon = 0.9 # probability for exploration to update Q-table
    decay_rate= 0.epsilon / (num_episodes/2)
    
    # training
    for episode in range(num_episodes):
        state = env.reset() # create a new instance of taxi, and get the initial state
        
        done = False
        
        while not done:
            action = selectAction(state)
                
            new_state, reward, done, _ = env.step(action) # take action and observe reward
        
            #Q-learning
            # Q(s,a) := Q(s,a) + learning_rate * (reward + discount_rate * max Q(s',a') - Q(s,a))
            qtable[state,action] = qtable[state,action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state,:])-qtable[state,action])
        
            state = new_state # update new state
        
        
        epsilon = max(0.1, epsilon-decay_rate) # epsilon decreases exponentially so the agent explores less over time
    print("Training complete")
    print(qtable)


if __name__== "__main__":
    env = gym.make('Taxi-v3')  # create Taxi environment
    
    # initialise qtable with 0s
    # stores the maximum expected future rewards the agent can expect for a certain action
    # row represents the state in the environment
    # column represents the action 
    # each cell is the Q-value for the state-action pair. 
    qtable = np.zeros([env.observation_space.n, env.action_space.n])
   
    train(env, qtable)
