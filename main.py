import gymnasium as gym
import panda_gym
from ddpg import Agent
env = gym.make("PandaReachDense-v3",render_mode="rgb_array")
obs_shape = env.observation_space['observation'].shape[0] + \
            env.observation_space['achieved_goal'].shape[0] + \
            env.observation_space['desired_goal'].shape[0] 
panda_agent = Agent(env=env, input_dims=obs_shape, agent_name='agent', save_in='Models/agent/')

score_history, avg_score_history = panda_agent.train(n_episodes=500,
                                                    plot_save='Results/DDPG/panda_agent Performance.png')
panda_agent.save_model()
reward = panda_agent.test_model(env=env, steps=100, render_save='Results/DDPG/panda_agent Policy', fps=5)
print('Reward: ', reward)