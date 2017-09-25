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
REPETITION_NUM = 3
gamma = 0.99
SOFT_UPDATE = 1e-3
ALR = 1e-4    
CLR = 1e-3

train_flag = False
train_int_cnt = 0
epi_flag = False
epi_int_cnt = 0
epi_cnt = 0

# env = gym.make('HalfCheetah-v1')
env=gym.make('Swimmer-v1')
state=env.reset()
output_shape = env.action_space.shape
input_shape = env.observation_space.shape     
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
print('Action size is', str(action_size))
print('Observation size is', state_size)
mujoco_mem_store = MMStore(MEM_SIZE)
mujoco_mem_store.ratio_fill(env, 0.1, MAX_EPI_INT)

sess = tf.Session()
K.set_session(sess)

actor = ActorNetwork(sess, state_size, action_size, BATCH_SIZE, SOFT_UPDATE, ALR)
actor.copy()
print('========== ACTOR ONLINE NETWORK ==========')
print(actor.online_nn.summary())
print('========== ACTOR TARGET NETWORK ==========')
print(actor.target_nn.summary())

critic = CriticNetwork(sess, state_size, action_size, BATCH_SIZE, SOFT_UPDATE, CLR)
critic.copy()
print('========== CRITIC ONLINE NETWORK ==========')
print(critic.online_nn.summary())
print('========== CRITIC TARGET NETWORK ==========')
print(critic.target_nn.summary())



## Training.
eval_accu_reward = 0
eval_flag = True
eval_state = env.reset()
eval_int_cnt = 0

actor.online_nn.save_weights('swimmer_online_actor_0.h5')

while train_flag==False:
    state = env.reset()
    epi_int_cnt=0
    epi_flag = False
    while epi_flag == False:
        #actor predict action
        action = np.zeros([1, action_size])
        noise = np.zeros([1, action_size])
        origin_action = actor.online_nn.predict(state.reshape(1, state_size))
        for i in range(action_size):
            action[0,i] = noise_func.function(origin_action[0,i],  0.0 , 0.15, 0.3) + origin_action[0, i]
        
        #enviroment interaction and process
        next_state, reward, done, info = env.step(action)
        mujoco_mem_store.append(state, action, reward, done, info, next_state)
        state = next_state
        epi_int_cnt = epi_int_cnt + 1
        train_int_cnt=train_int_cnt + 1

        #batch sample and train 
        y = np.zeros((BATCH_SIZE, 1))
        if epi_cnt % 2 == 0:
            states, actions, rewards, dones, infos, next_states = mujoco_mem_store.sample_dataSelection(BATCH_SIZE, state_size, action_size, REPETITION_NUM)
        else:
            states, actions, rewards, dones, infos, next_states = mujoco_mem_store.sample(BATCH_SIZE, state_size, action_size)
        # states, actions, rewards, dones, infos, next_states = mujoco_mem_store.sample_dataSelection(BATCH_SIZE, state_size, action_size, REPETITION_NUM)
        # states, actions, rewards, dones, infos, next_states = mujoco_mem_store.sample(BATCH_SIZE, state_size, action_size)
        target_q_values = critic.target_nn.predict_on_batch([next_states, actor.target_nn.predict(next_states)])
        for i in range(BATCH_SIZE):
            if dones[i]:
                y[i] = rewards[i]
            else:
                y[i] = rewards[i] + gamma*target_q_values[i]
        loss = critic.online_nn.train_on_batch([states,actions], y)              
        actions_for_grad = actor.online_nn.predict_on_batch(states)
        grads = critic.action_gradients(states, actions_for_grad)
        actor.train(states, grads)
        actor.soft_target_nn_update()
        critic.soft_target_nn_update()

        if(epi_int_cnt>MAX_EPI_INT)|(done==True):
            epi_flag = True
    epi_cnt = epi_cnt + 1
    print(epi_cnt)
    if epi_cnt%5 == 0:
        eval_accu_reward = 0
        eval_flag = True
        eval_state = env.reset()
        eval_int_cnt = 0
        while(eval_flag == True):
            #env.render()
            eval_action = actor.online_nn.predict(eval_state.reshape(1,state_size)) 
            eval_state, reward, done, info=env.step(eval_action)
            eval_accu_reward = eval_accu_reward+reward
            eval_int_cnt = eval_int_cnt + 1
            if(eval_int_cnt>MAX_EPI_INT)|( done==True):
                eval_flag = False
        print('Evaluation Result:',eval_accu_reward,eval_int_cnt)
        actor.online_nn.save_weights('swimmer_online_actor_'+str(epi_cnt)+'.h5')
        # actor.target_nn.save_weights('half_target_actor_'+str(epi_cnt)+'.h5')
    if train_int_cnt > MAX_INTERACTION:
        train_flag = True



