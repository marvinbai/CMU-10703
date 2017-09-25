import gym
import numpy as np
import random
import argparse
from gym import wrappers
from keras.models import model_from_json, Model
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
MAX_EPI_INT = 500
MEM_SIZE = 500
BATCH_SIZE = 64
gamma = 0.99
SOFT_UPDATE = 1e-3
ALR = 1e-4    
CLR = 1e-3
REPEAT = 10


env = gym.make('HalfCheetah-v1')
eval_state = env.reset()
output_shape=env.action_space.shape
input_shape=env.observation_space.shape     
state_size=env.observation_space.shape[0]
action_size=env.action_space.shape[0]


sess=tf.Session()
K.set_session(sess)
actor = ActorNetwork(sess, state_size, action_size, BATCH_SIZE, SOFT_UPDATE, ALR)
actor.copy()
critic = CriticNetwork(sess, state_size, action_size, BATCH_SIZE, SOFT_UPDATE, CLR)
critic.copy()

# env=gym.make('Swimmer-v1')
# env = wrappers.Monitor(env, '/video')
reward_total = 0
f = open('LearningCurve.txt','w')

for cnt_weight in range(0, 275, 5):

    for i in range(REPEAT):
        train_flag = False
        train_int_cnt = 0
        epi_flag = False
        epi_int_cnt = 0
        epi_cnt = 0
        
        eval_state = env.reset()

        eval_accu_reward=0
        eval_flag=True
        eval_int_cnt=0

        # Load weight.
        actor.online_nn.load_weights('half_online_actor_' + str(cnt_weight) + '.h5')

        # Make video.

        while(eval_flag==True):
            # env.render()
            # time.sleep(0.1)
            eval_action = actor.online_nn.predict(eval_state.reshape(1,state_size)) 
            eval_state, reward, done, info = env.step(eval_action)
            #print(eval_action)
            #print(reward)
            #print(done)
            #print(eval_int_cnt)
            eval_int_cnt = eval_int_cnt + 1
            eval_accu_reward=eval_accu_reward+reward
            if (eval_int_cnt>MAX_EPI_INT)|(done==True):
               eval_flag = False
        reward_total = reward_total + eval_accu_reward
        # print(eval_accu_reward)
    reward_total = reward_total / REPEAT    
           
    f.write(str(cnt_weight) + '\t' + str(reward_total) + '\n')


f.close()
