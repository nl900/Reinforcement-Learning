import gym
import numpy as np


"""
SARSA is acronym for State Action Reward State Action
as the value function is learned according to the action derived from 
the current policy used (on-policy)
"""

def train(env, qtable):
    # hyperparameters
    epsilon = 0.9
    learning_rate = 0.85
    discount_rate = 0.95
    
    # training variables
    total_episodes = 1000
    max_steps = 100
    
    def selectAction(state):
        if np.random.uniform(0, 1) < epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(qtable[state, :])
        
    
    for episode in range(total_episodes):
        state = env.reset()
        action = selectAction(state) # initial action
            
        done = False
        while not done:
            
            new_state, reward, done, info = env.step(action) # apply action to next state
            new_action = selectAction(new_state)
            
            # SARSA update
            # Q(s,a) := Q(s,a) + learning_rate * (reward + discount_rate * Q(s',a') - Q(s,a))
            predict = qtable[state, action]
            target = reward + discount_rate * qtable[new_state, new_action]
            qtable[state, action] = learning_rate * (target - predict)
            
            if done: break
                
            state, action = new_state, new_action
            
    env.close()
    
    print("Trained Q Table")
    print(qtable)
          
            
    
if __name__== "__main__":
    env = gym.make('FrozenLake-v1')
    qtable = np.zeros((env.observation_space.n, env.action_space.n)) # initialise Q table
    train(env, qtable)
