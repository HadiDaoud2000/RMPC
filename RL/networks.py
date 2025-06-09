import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#defining Actor neural network 
class Actor(nn.Module): #inherit from class torch.nn.module
    #defining constructor
    def __init__(self, input_state_shape, n_actions1, network_name, checkpoints_dir="Data"):
        super(Actor, self).__init__() #superclass:used to call the __init__ method of the parent class of Actor
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        self.checkpoints_file = os.path.join(checkpoints_dir, network_name + ".pth")
        
        self.input_dim = input_state_shape
        self.output_dim = n_actions1
        #Network Architecture:
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out_action = nn.Linear(256, self.output_dim)
    
    def forward(self, state):
        x = T.relu(self.fc1(state))
        x = T.relu(self.fc2(x))
        action = T.tanh(self.out_action(x)) #to constrain the output to [-1, 1]
        return action
    
class Critic(nn.Module): #inherit from class torch.nn.module
    #defining constructor
    def __init__(self, input_action_shape, network_name, checkpoints_dir="Data"):
        super(Critic, self).__init__() #superclass:used to call the __init__ method of the parent class of Actor
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        self.checkpoints_file = os.path.join(checkpoints_dir, network_name + ".pth")
        
        self.input_dim = input_action_shape
        #Network Architecture:
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.Q = nn.Linear(256, 1)
    
    def forward(self, state, action):
        x = T.cat([state, action], dim=1) #to concatenates state, and action tensors horizontally
        #The concatenation step allows the Critic to make predictions based on the full context
        # (state + action) rather than just the state alon
        x = T.relu(self.fc1(x))
        x = T.relu(self.fc2(x))
        Q = self.Q(x)
        return Q    