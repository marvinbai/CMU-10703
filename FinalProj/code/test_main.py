import gym
import numpy as np
import copy
from numpy import array
from PIL import Image

from improcess import AtariProcessor
from improcess import HistoryStore
from policy import GreedyPolicy
from policy import UniformRandomPolicy
from memhelpers import NNMemStore
IMAGE_SIZE = (84,84)
HISTORY_LENGTH = 4

MEM_SIZE=2000
INIT_MEM_RATIO=0.5

env = gym.make('BreakoutDeterministic-v0')
observation=env.reset()
num_actions = env.action_space.n

atari_processor = AtariProcessor(IMAGE_SIZE)
history_store = HistoryStore(HISTORY_LENGTH, IMAGE_SIZE)
greedy_selector = GreedyPolicy()
random_selector=UniformRandomPolicy(num_actions)
episode_end_flag=False
mem_store=NNMemStore(MEM_SIZE,(84,84,4))
observation=env.reset()
state=atari_processor.state_for_mem(observation)
history_store.add_history(state)
i=0
life=False
first_step=True
while episode_end_flag==False:
      nn_input=history_store.get_history()
      action=random_selector.select_action()
      observation, reward, done, info = env.step(action)
      episode_end_flag=done
      state=atari_processor.state_for_mem(observation)
      history_store.add_history(state)
      next_nn_input=history_store.get_history()
      if(first_step==True):
          life=False
          first_step=False
          prev_info=info
      else:
          if(info!=prev_info):
              #print(info)
              life=True
              #print(life)
              #print(done)
          else:
              life=False
          prev_info=info
      reward=atari_processor.process_reward(reward)
      mem_store.append(nn_input, action, reward, next_nn_input,done,life)
      mem_nn_input,mem_action,mem_reward, mem_next_nn_input,mem_done,mem_life=mem_store.get_last_sample() 
      #write image to file
      im=Image.fromarray(state)
      im.save("./env_images/"+str(i), "GIF")
      hist_im=Image.fromarray(next_nn_input[:,:,3])
      hist_im.save("./history_images/"+str(i), "GIF")
      flat_history_im=Image.fromarray(history_store.flat(next_nn_input))
      flat_history_im.save("./history_images/flat"+str(i),"GIF")

      mem_im_0=Image.fromarray(mem_next_nn_input[:,:,0])
      mem_im_1=Image.fromarray(mem_next_nn_input[:,:,1])
      mem_im_2=Image.fromarray(mem_next_nn_input[:,:,2])
      mem_im_3=Image.fromarray(mem_next_nn_input[:,:,3])

      mem_im_0.save("./mem_images/"+str(i)+str(0), "GIF")
      mem_im_1.save("./mem_images/"+str(i)+str(1), "GIF")
      mem_im_2.save("./mem_images/"+str(i)+str(2), "GIF")
      mem_im_3.save("./mem_images/"+str(i)+str(3), "GIF")
      i=i+1
      

print(mem_store.get_status())
print(mem_store.get_mem_length())
print(mem_store.find_episode_end_position())
print(mem_store.find_life_lost_position())
print(mem_store.find_reward_position())
