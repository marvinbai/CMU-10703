import gym
import numpy as np
import random
import argparse

from keras.models import Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
#from keras.engine.training import collect_trainable_weights
from keras import backend as K
from mmstore import MMStore
from mujoco_actor_nn import ActorNetwork
from mujoco_critic_nn import CriticNetwork
from OU import OU
import time

noise_func = OU()
MAX_INTERACTION = 10000000
MAX_EPI_INT = 100
MEM_SIZE = 500000
BATCH_SIZE = 64
REPETITION_NUM = 2
gamma = 0.99
SOFT_UPDATE = 1e-3
ALR = 1e-4    
CLR = 1e-3

train_flag = False
train_int_cnt = 0
epi_flag = False
epi_int_cnt = 0
epi_cnt = 0

env = gym.make('HalfCheetah-v1')
#env=gym.make('Swimmer-v1')
state=env.reset()
output_shape = env.action_space.shape
input_shape = env.observation_space.shape     
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
print('Action size is', str(action_size))
print('Observation size is', state_size)
mujoco_mem_store = MMStore(MEM_SIZE)
mujoco_mem_store.ratio_fill(env, 0.1, MAX_EPI_INT)

f = open('DataSelection.txt','w')
states, actions, rewards, dones, infos, next_states = mujoco_mem_store.sample_dataSelection(BATCH_SIZE, state_size, action_size, REPETITION_NUM)
for i in range(rewards.shape[0]):
    f.write(str(rewards[i][0]))
    f.write('\n')
f.close()

f = open('DataNoSelection.txt','w')
states, actions, rewards, dones, infos, next_states = mujoco_mem_store.sample(BATCH_SIZE, state_size, action_size)
for i in range(rewards.shape[0]):
    f.write(str(rewards[i][0]))
    f.write('\n')
f.close()
