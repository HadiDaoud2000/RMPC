import numpy as np
class Replay_buffer:
    #initializing the cicular buffer
    def __init__(self, max_size, input_state_shape, n_actions):
        self.max_size = max_size #size of the buffer
        self.LEI = 0 #last element index
        self.action_mem = np.zeros((max_size, n_actions)) #action memory
        self.reward_mem = np.zeros((max_size)) #reward memory
        self.state_mem = np.zeros((max_size, input_state_shape)) #state memory
        self.new_state_mem = np.zeros((max_size, input_state_shape)) #new state memory
        self.flag_mem = np.zeros((max_size)) #done flag memory
    
    #storing data in the buffer 
    def store(self, current_state, action, reward, new_state, done_flag):
        i = self.LEI % self.max_size #dynamic index to ensure saving newest data
        self.state_mem[i] = current_state
        self.action_mem[i] = action
        self.reward_mem[i] = reward
        self.new_state_mem[i] = new_state
        self.flag_mem[i] = done_flag
        self.LEI += 1
    
    
    #randomly choosing a batch of experiences 
    def random_sample(self, batch_size):
        #we used min(self.max_size, self.LEI) to insure taking valid experiences 
        #replace=False means that each index appears at most once in the batch
        batch = np.random.choice(min(self.max_size, self.LEI), batch_size, replace=False)
        states = self.state_mem[batch]
        new_states = self.new_state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        done_flags = self.flag_mem[batch]
        
        return states, new_states, actions, rewards, done_flags
        
        