import numpy as np
import gymnasium as gym
import panda_gym
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
from IPython import display
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from networks import Actor, Critic
#from replay import ExperienceReplayMemory
from replay_buffer import Replay_buffer

class Agent:
    def __init__(self, env, input_dims, alpha=0.001, beta=0.002, gamma=0.99, tau=0.05,
                 batch_size=256, replay_buffer_size=1000000, noise_factor=0.1,
                 agent_name='agent', save_in=None, load_from=None):
    # hyperparameters
        self.alpha = alpha  # actor learning rate
        self.beta = beta    # critic learning rate
        self.gamma = gamma  # discount factor
        self.tau = tau      # soft update factor
        self.batch_size = batch_size  # training batch size
        self.input_dims = input_dims
        self.step_count = 0
        self.noise_factor = noise_factor   # exploration noise factor
        self.device = 'cpu' 
        self.agent_name = agent_name
        self.training_flag = False
        if save_in is None:
            self.save_in = f'Data/{agent_name}'
        else:
            self.save_in = save_in        
        self.env = env
        self.n_actions = env.action_space.shape[0]
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.buffer_mem = Replay_buffer(replay_buffer_size, input_dims, self.n_actions)
        if load_from:
            self.build_neural_network(self.n_actions, checkpoints_dir=load_from)
            self.load_model()
        else:
            self.build_neural_network(self.n_actions)
            # update weights of the target actor
            self.par_update(self.target_actor, self.actor, self.tau)

            # update weights of the first target critic network
            self.par_update(self.target_critic, self.critic, self.tau) 
            
    def select_action(self, observation, train=True):

        if self.training_flag: #check training flag
            train = False
        
        state = T.tensor([observation], dtype=T.float32).to(self.device) #prepare the state
        action = self.actor(state).detach().numpy()[0] #Actor Network Prediction
        #detach Prevents gradient computation for this operation
        
        if train:
            action += np.random.normal(loc=0, scale=self.noise_factor, size=self.n_actions) #adding noise
        
        action = np.clip(action, self.min_action, self.max_action) #Ensures the action lies within the permissible range

        return action
        
    def build_neural_network(self, n_actions, checkpoints_dir=None):
        if checkpoints_dir is None:
            checkpoints_dir=self.save_in
        self.actor = Actor(input_state_shape=self.input_dims, n_actions1=n_actions, 
                           network_name="actor", checkpoints_dir=checkpoints_dir).to(self.device)
        self.critic = Critic(input_action_shape=self.input_dims+self.n_actions,
                               network_name="critic", checkpoints_dir=checkpoints_dir).to(self.device)

        self.target_actor = Actor(input_state_shape=self.input_dims, n_actions1=n_actions, 
                                  network_name="target_actor", checkpoints_dir=checkpoints_dir).to(self.device)
        self.target_critic = Critic(input_action_shape=self.input_dims+self.n_actions, 
                                      network_name="target_critic", checkpoints_dir=checkpoints_dir).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.alpha)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.beta)

        self.target_actor_optimizer = optim.Adam(self.target_actor.parameters(), lr=self.alpha)
        self.target_critic_optimizer = optim.Adam(self.target_critic.parameters(), lr=self.beta)    
    
    def par_update(self, new_network, old_network, tau):
        new_parameters = new_network.state_dict()
        old_parameters = old_network.state_dict()
        for i in old_parameters:
            new_parameters[i] = tau * old_parameters[i] + (1.0 - tau) * new_parameters[i]
        new_network.load_state_dict(new_parameters)
        
          
    def train(self, n_episodes, plot_save=None):
        env = self.env
        score_history = [] #keeps track of episode rewards
        avg_score_history = [] #stores the average score of the last 100 game
        for i in tqdm(range(n_episodes), desc='Training..'): #training loop 
            #reset training parameters
            done = False
            truncated = False
            score = 0
            step = 0
            obs_arr = []
            actions_arr = []
            new_obs_arr = []
            obs, info = env.reset() #reset the environment ang get the initial observation
            while True: #episode interactions loop 
                current_state_obs, achieved_goal, desired_goal = obs.values() #getting parameters from observation
                current_state = np.concatenate((current_state_obs, achieved_goal, desired_goal)) #combine those parameters as the current state
                action = self.select_action(current_state) #select an action depending on the current state and agents policy
                new_observation, reward, done, truncated, _ =env.step(np.array(action)) #take the action to reach the new state and transitional variables 
                new_obs, new_ach_goal, new_des_goal = new_observation.values() #getting parameters from new observation
                new_state = np.concatenate((new_obs, new_ach_goal, new_des_goal)) #combine the parameters to get the new state
                
                self.buffer_mem.store(current_state, action, reward, new_state, done) #save training parameters in replay buffer memory
                obs_arr.append(obs) #saving old observations for the same episode for all its time steps
                actions_arr.append(action) #saving actions for the same episode for all its time steps
                new_obs_arr.append(new_observation) #saving new observations for the same episode for all its time steps
                obs = new_observation #update current observation
                score += reward #accumulative reward during episode
                step += 1 
                if (done or truncated):
                    break
            self.HER(obs_arr, actions_arr, new_obs_arr)    
            for j in range(64):
                self.optimize_model() 
        
          
            score_history.append(score) #saving accumulative reward for each episode
            avg_score = np.mean(score_history[-100:]) #average accumulative reward for last 100 episode
            avg_score_history.append(avg_score) #saving average accumulative reward for last 100 episode
        
                
            if self.save_in and i % (n_episodes//10)==0: #saving model each 10 % of total episodes number
                self.save_model()    
                    
        self.plot_scores(scores=score_history, avg_scores=avg_score_history, plot_save=plot_save)
        return score_history, avg_score_history  
    
    
    def plot_scores(self, scores, avg_scores,plot_save):
        plt.figure(figsize=(12,10))
        plt.plot(scores)
        plt.plot(avg_scores)
        plt.xlabel('Episode')
        plt.ylabel('Score')
        
    def test_model(self, steps, env=None, save_states=False, render_save=None, fps=30):
        if env is None:
            env = self.env
        episode_score = 0
        state_list = []     # List to store state feature vectors
        
        observation, info = env.reset()
        current_observation, current_achieved_goal, current_desired_goal = observation.values()
        state = np.concatenate((current_observation, current_achieved_goal, current_desired_goal))
                        
        if save_states:
            state_list.append(T.tensor(state, dtype=T.float32).unsqueeze(0).to(self.device))
        
        images = []
        done = False
        truncated = False
        
        with T.inference_mode():
            for i in range(steps):
                if render_save:
                    images.append(env.render())

                action = self.select_action(state)

                observation, reward, done, truncated, _ = env.step(np.array(action))
                
                current_observation, current_achieved_goal, current_desired_goal = observation.values()
                state = np.concatenate((current_observation, current_achieved_goal, current_desired_goal))

                if save_states:
                    state_list.append(T.tensor(state, dtype=T.float32).unsqueeze(0).to(self.device))
                
                episode_score += reward

                if done or truncated:
                    if render_save:
                        images.append(env.render())
                    break

        if render_save:
            # env.close()
            imageio.mimsave(f'{render_save}.gif', images, fps=fps, loop=0)
            with open(f'{render_save}.gif', 'rb') as f:
                display.display(display.Image(data=f.read(), format='gif'))
                
        if not save_states:
            return episode_score
        else:
            return episode_score, state_list            
                
    def save_model(self):
        T.save(self.actor.state_dict(), self.actor.checkpoints_file)
        T.save(self.critic.state_dict(), self.critic.checkpoints_file)
        T.save(self.target_actor.state_dict(), self.target_actor.checkpoints_file)
        T.save(self.target_critic.state_dict(), self.target_critic.checkpoints_file)

    def load_model(self):

        self.training_flag = True
        self.actor.load_state_dict(T.load(self.actor.checkpoints_file))
        self.critic.load_state_dict(T.load(self.critic.checkpoints_file))
        self.target_actor.load_state_dict(T.load(self.target_actor.checkpoints_file))
        self.target_critic.load_state_dict(T.load(self.target_critic.checkpoints_file))


    def optimize_model(self):
        # check if there are enough experiences in memory
        if self.buffer_mem.LEI < self.batch_size:
            return

        # sample a random batch of experiences from memory
        states, new_states, actions, rewards, done_flags = self.buffer_mem.random_sample(self.batch_size)

        states = T.tensor(states, dtype=T.float32).to(self.device)
        next_states = T.tensor(new_states, dtype=T.float32).to(self.device)
        actions = T.tensor(actions, dtype=T.float32).to(self.device)
        rewards = T.tensor(rewards, dtype=T.float32).to(self.device)
        done_flags = T.tensor(done_flags, dtype=T.float32).to(self.device)

        # calculate critic network loss
        target_actions = self.target_actor(next_states) #Use the target actor network 
        #target_actor to predict the next actions for next_states
        new_critic_value = self.target_critic(next_states, target_actions).squeeze(1) #evaluate target critic
        target = rewards + self.gamma * new_critic_value * (1 - done_flags) #calculate target values using Bellman equation
        #Target=Reward+γ×Next Critic Value×(1−Done Flag)
        Q_value= self.critic(states, actions).squeeze(1) #predicting Q value
        critic_loss = F.mse_loss(target, Q_value) #Critic Loss=MSE(Target,Predicted Q Value)

        self.critic_optimizer.zero_grad() #Clear any gradients accumulated in previous iterations.
        critic_loss.backward() # Compute gradients of the critic loss with respect to the critic network parameters.
        self.critic_optimizer.step() #Update the critic network parameters using the optimizer

        # Calculate actor network loss
        actor_loss = -self.critic(states, self.actor(states)).mean()

        # Apply gradient descent with the calculated actor loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        # update weights of the target actor
        self.par_update(self.target_actor, self.actor, self.tau)

        # update weights of the target critic network
        self.par_update(self.target_critic, self.critic, self.tau)
        
    def HER(self, observations, actions, new_observations, k = 4):
        
        # augment the replay buffer
        num_samples = len(actions)
        for index in range(num_samples):
            for _ in range(k):
                # sample a future state (observation and goal) 
                future_index = np.random.randint(index, num_samples)
                future_observation, future_achieved_goal, _ = new_observations[future_index].values()
                # extract current observation and action from the experience
                observation, _, _ = observations[future_index].values()
                
                # create state representation 
                state = T.tensor(np.concatenate((observation, future_achieved_goal, future_achieved_goal)), 
                                     dtype=T.float32).to(self.device)

                new_observation, _, _ = new_observations[future_index].values()
                
                # create new state representation 
                new_state = T.tensor(np.concatenate((new_observation, future_achieved_goal, 
                                                          future_achieved_goal)), dtype=T.float32).to(self.device)

                # extract action from the experience
                action = T.tensor(actions[future_index], dtype=T.float32).to(self.device)
                
                # calculate reward based on achieving the future goal from the current state and action
                reward = self.env.unwrapped.compute_reward(future_achieved_goal, future_achieved_goal, 1.0)

                # store augmented experience in buffer
                state = state.cpu().numpy()
                action = action.cpu().numpy()
                new_state = new_state.cpu().numpy()

                self.buffer_mem.store(state, action, reward, new_state, True)