import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from collections import namedtuple
from itertools import count

import psutil
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

env = gym.make('Blackjack-v0') # OpenAI environment
env.__init__(natural=True)
######### Replay Memory #########
# de https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

Transition = namedtuple('Transition', ('state', 'action', 'next_state','reward'))

class ReplayMemory(object):
    """Mémoire des dernières actions de taille capacity"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, *args):
        """ Ajoute un nouvel élément dans la mémoire s'il reste de la place.
            Si la capacité est atteinte, remplace l'élément le plus vieux"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """ Retourne un ensemble random de batch_size exemples """
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
    
######### DQN #########
class DQN(nn.Module):
    """ Réseau de neurones pour estimer la valeur Q"""
    
    def __init__(self, Hlayers, dropout):
        super().__init__()
        self.fc_input = nn.Linear(3, Hlayers[0])
        self.HLayers = Hlayers
        self.fc_ouput = nn.Linear(Hlayers[-1], 2)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        x = x.view(-1, 3)
        x = F.relu(self.fc_input(x))
        for i in range(len(self.HLayers)-1):
            fc = nn.Linear(self.HLayers[i], self.HLayers[i+1])
            x = self.drop(x)
            x = F.relu(fc(x))
        x = self.fc_ouput(x)
        return x



########## Input extraction ##########

env.reset()


######### Training #########

BATCH_SIZE = 100
GAMMA = 1

EPS_START = 1 #0.9
EPS_END = 0 #0.05
NUM_TRAIN = 800
EPS_K = 5
EPS_F = 800

TARGET_UPDATE = 10
learning_rate = 0.001 #0.0001


# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN([30,20,10], 0.5)
target_net = DQN([30,20,10], 0.5)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=learning_rate)
memory = ReplayMemory(100)

def test_model():
    rewards = []
    for i in range(10000):
         # Jouer la partie
         state = torch.from_numpy(np.ascontiguousarray(env.reset(), dtype=np.float32)).to(device)
         for t in count():
             # Selectionner et faire l'action
             action = target_net(state).max(1)[1].view(1, 1)
             next_state, reward, done, _ = env.step(action.item())
             
             if done:
                 rewards.append(reward)
                 break
             next_state = torch.from_numpy(np.ascontiguousarray(next_state, dtype=np.float32)).to(device)
             # Passe au prochain état
             state = next_state
    
    count_win = sum(1 for x in rewards if x>=1)
    count_lo = sum(1 for x in rewards if x==-1)
    count_d = sum(1 for x in rewards if x==0)
    count_bj = sum(1 for x in rewards if x==1.5)
    
    return(count_win, count_lo, count_d, count_bj)

def select_action(state, epoque, methode):
    
    if methode==0 : # Diminution linéraire de epsilon
        if (epoque < 0.3*NUM_TRAIN):
            n = 0.3*NUM_TRAIN
            eps_f = 0.9*EPS_START
            a = (eps_f - EPS_START)/n
            eps_threshold = EPS_START + epoque*a
        elif (epoque < 0.7*NUM_TRAIN):
            n = 0.4*NUM_TRAIN
            eps_i = 0.9*EPS_START
            eps_f = 0.1*EPS_START
            a = (eps_f - eps_i)/n
            eps_threshold = eps_i + epoque*a
        elif (epoque < NUM_TRAIN):
            n = 0.3*NUM_TRAIN
            eps_i = 0.1*EPS_START
            a = (EPS_END - eps_i)/n
            eps_threshold = eps_i + epoque*a
        else:
            eps_threshold = 0
    else: #diminution exponentielle de epsilon
        if (epoque > EPS_F):
            eps_threshold = 0
        else:
            eps_threshold = EPS_START + (EPS_END - EPS_START) * math.exp((epoque-EPS_F)/EPS_K)
        
    
    sample = random.random()
    
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


#episode_durations = []
rewards = []
losses = []
used_memory = []
used_cpu = []

def plot_losses():
    plt.figure(1)
    plt.clf()
    losses_t = torch.tensor(losses, dtype=torch.float)
    plt.title('Pertes')
    plt.xlabel('Epoque')
    plt.ylabel('Perte')
    plt.plot(losses_t.numpy())
    # Moyenne
    if len(losses_t) > 1:
        m = min(len(losses_t), 50)
        means = losses_t.unfold(0,m,1).mean(1).view(-1)
        means = torch.cat((losses_t[0:m-1], means))
        plt.plot(means.numpy())
    
    plt.legend(['Perte', 'Perte moyenne'])
    
    plt.pause(0.001)
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def plot_rewards():
    plt.figure(2)
    plt.clf()
    rewards_t = torch.tensor(rewards, dtype=torch.float)
    plt.title('Rewards')
    plt.xlabel('Epoque')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())
    
    # Moyenne
    if len(rewards_t) > 1:
        m = min(len(rewards_t), 50)
        means = rewards_t.unfold(0,m,1).mean(1).view(-1)
        means = torch.cat((rewards_t[0:m-1], means))
        plt.plot(means.numpy())
    
    plt.legend(['Rewards', 'Moyenne'])
    
    plt.pause(0.001)
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def plot_ressources():
    # Memory
    plt.figure(3)
    plt.clf()
    memory_t = torch.tensor(used_memory, dtype=torch.float)
    plt.title('Used memory...')
    plt.xlabel('Episode')
    plt.ylabel('Memory (MB)')
    plt.plot(memory_t.numpy())
    plt.pause(0.001)
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
    
    #CPU
    plt.figure(4)
    plt.clf()
    cpu_t = torch.tensor(used_cpu, dtype=torch.float)
    plt.title('Used CPU...')
    plt.xlabel('Episode')
    plt.ylabel('Usage (%)')
    plt.plot(cpu_t.numpy())
    
    plt.pause(0.001)
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    if (non_final_mask.max().numpy() > 0):
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    if (non_final_mask.max().numpy() > 0):
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    losses.append(loss)
    plot_losses()
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


process = psutil.Process(os.getpid())
start_time = time.time() 
num_epoques = 1000
num_parties = 100
for i_epoque in range(num_epoques):
    # Pour chaque epoque, faire 100 parties et entrainer
    cum_reward = 0
    
    for i_partie in range(num_parties):
        # Commencer la partie
        state = torch.from_numpy(np.ascontiguousarray(env.reset(), dtype=np.float32)).to(device)
        
        # Jouer la partie
        for t in count():
            # Selectionner et faire l'action
            action = select_action(state, i_epoque, 0)
            next_state, reward, done, _ = env.step(action.item())
            
            reward = torch.tensor([reward], device=device, dtype=torch.float)
            
            next_state = torch.from_numpy(np.ascontiguousarray(next_state, dtype=np.float32)).to(device)
            # Si fin de la partie, next_state = None
            if done:
                next_state = None
    
            # Enregistre dans la mémoire
            memory.push(state, action, next_state, reward)
    
            # Passe au prochain état
            state = next_state
            
            # Si fin de la partie, passer à la prochaine partie
            if done:
                break
        
        cum_reward += reward
    
    rewards.append(cum_reward)
    plot_rewards()
    # entraine le modèle (target network)
    optimize_model()  
    
#    used_memory.append((process.memory_info().rss)/(1024*1024))
#    used_cpu.append(process.cpu_percent())
#    plot_ressources()
    
    # Update the target network, copying all weights and biases in DQN
    if i_epoque % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

elapsed_time = time.time() - start_time

print('Complete')
print("Temps écoulé : %d s" % elapsed_time)
print("Reward cummulatif : %d" % np.sum(rewards))
print()
rewards_t = torch.tensor(rewards, dtype=torch.float)
means = rewards_t.unfold(0,50,1).mean(1).view(-1)
print("Rewards moyen (50 derniers) : %2.2f" % means[-1])

losses_t = torch.tensor(losses, dtype=torch.float)
means = losses_t.unfold(0,50,1).mean(1).view(-1)
print("Pertes moyenne (50 derniers) : %2.2f" % means[-1])

count_win, count_lo, count_d, count_bj = test_model()
print("Win: %d \nPerdu: %d \nNul : %d \nBJ : %d" % (count_win,count_lo,count_d,count_bj))

env.close()
plt.ioff()
plt.show()
