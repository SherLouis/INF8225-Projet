import random
import numpy as np

#Tiré de : https://github.com/pradhyo/blackjack


class AgentQ():
    def __init__(self,env,epsilon=1.0,ep_start=1.0,ep_end=0.0,scaling_factor=1.0,alpha=0.5,gamma=0.9,num_episodes_to_train=800,agressive_e_greedy=False, random=False):
        self.env = env

        # n is number of valid actions from the souce code
        self.valid_actions = list(range(self.env.action_space.n))

        # Set parameters of the learning agent
        self.Q = dict()                     # Q-table which will be a dictionary of tuples
        self.epsilon = epsilon              # Random exploration factor
        self.epsilon_start=ep_start         # Starting epsilon for agressive e-greedy policy
        self.epsilon_end=ep_end             # Ending epsilon for agressive e-greedy policy
        self.scaling_factor=scaling_factor  # Scaling factor for agressive e-greedy policy
        self.alpha = alpha                  # Learning factor
        self.gamma = gamma                  # Discount factor- closer to 1 learns well into distant future

        # epsilon will reduce linearly until it reaches 0 based on num_episodes_to_train
        # epsilon drops to 90% of its inital value in the first 30% of num_episodes_to_train
        # epsilon then drops to 10% of its initial value in the next 40% of num_episodes_to_train
        # epsilon finally becomes 0 in the final 30% of num_episodes_to_train

        self.num_episodes_to_train = num_episodes_to_train # Change epsilon each episode based on this
        self.small_decrement = (0.1 * epsilon) / (0.3 * num_episodes_to_train) # reduces epsilon slowly
        self.big_decrement = (0.8 * epsilon) / (0.4 * num_episodes_to_train) # reduces epilon faster

        #Determine which type of epsilon greedy policy is applied

        self.agressive_e_greedy=agressive_e_greedy
        self.num_episodes_to_train_left = num_episodes_to_train
        self.current_episode=1

        #Determine if epsilon always equals 1(random actions)
        self.random=random
   
        
    def update_parameters(self):
        """
        Update epsilon and alpha after each action
        Set them to 0 if not learning
        Epsilon equals 1 if the agent plays randomly
        """

        if(not self.random and not self.agressive_e_greedy):
            if self.num_episodes_to_train_left > 0.7 * self.num_episodes_to_train:
                self.epsilon -= self.small_decrement
            elif self.num_episodes_to_train_left > 0.3 * self.num_episodes_to_train:
                self.epsilon -= self.big_decrement
            elif self.num_episodes_to_train_left > 0:
                self.epsilon -= self.small_decrement
            else:
                self.epsilon = 0.0
                self.alpha = 0.0

            self.num_episodes_to_train_left -= 1

        elif(self.agressive_e_greedy):
            if(self.current_episode<=self.num_episodes_to_train):
                self.epsilon=self.epsilon_start+(self.epsilon_end-self.epsilon_start)*np.exp((self.current_episode-self.num_episodes_to_train)/self.scaling_factor)
            
            self.current_episode+=1

    def create_Q_if_new_observation(self, observation):
        """
        Set intial Q values to 0.0 if observation not already in Q table
        """
        if observation not in self.Q:
            self.Q[observation] = dict((action, 0.0) for action in self.valid_actions)

    def get_maxQ(self, observation):
        """
        Called when the agent is asked to find the maximum Q-value of
        all actions based on the 'observation' the environment is in.
        """
        self.create_Q_if_new_observation(observation)
        return max(self.Q[observation].values())

    def choose_action(self, observation):
        """
        Choose which action to take, based on the observation.
        If observation is seen for the first time, initialize its Q values to 0.0
        """
        self.create_Q_if_new_observation(observation)

        # uniformly distributed random number > epsilon happens with probability 1-epsilon
        if random.random() > self.epsilon:
            maxQ = self.get_maxQ(observation)

            # multiple actions could have maxQ- pick one at random in that case
            # this is also the case when the Q value for this observation were just set to 0.0
            action = random.choice([k for k in self.Q[observation].keys()
                                    if self.Q[observation][k] == maxQ])
        else:
            action = random.choice(self.valid_actions)

        self.update_parameters()

        return action

    def play(self, observation):
        """
        Choose action, based on the observation and the Q-Table
        """
        self.create_Q_if_new_observation(observation)

        maxQ = self.get_maxQ(observation)

        # multiple actions could have maxQ- pick one at random in that case
        # this is also the case when the Q value for this observation were just set to 0.0
        action = random.choice([k for k in self.Q[observation].keys()
                                if self.Q[observation][k] == maxQ])

        return action

    def learn(self, observation, action, reward, next_observation):
        """
        Called after the agent completes an action and receives an award.
        This function does not consider future rewards
        when conducting learning.
        """

        # Q = Q*(1-alpha) + alpha(reward + discount * utility of next observation)
        # Q = Q - Q * alpha + alpha(reward + discount * self.get_maxQ(next_observation))
        # Q = Q - alpha (-Q + reward + discount * self.get_maxQ(next_observation))
        self.Q[observation][action] += self.alpha * (reward
                                                     + (self.gamma * self.get_maxQ(next_observation))
                                                     - self.Q[observation][action])
