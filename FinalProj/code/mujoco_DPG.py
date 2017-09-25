import gym
import numpy as np
import random
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


env = gym.make('HalfCheetah-v1')
prev_state=env.reset()
#num_actions = env.action_space.n
print(prev_state)
#print(num_actions)
print(env.action_space)
print(env.action_space.high)
print(env.action_space.low)
print(env.observation_space)
print(env.observation_space.high)
print(env.env.observation_space.low)
output_shape=env.action_space.shape
input_shape=env.observation_space.shape
print(input_shape)
state_size=input_shape[0]
print(output_shape)
action_size=output_shape[0]
BATCH_SIZE=0
soft_update=0.001
lr=0.001

sess = K.get_session()
actor=ActorNetwork(sess, state_size, action_size, BATCH_SIZE, soft_update, lr)
model, trainable_weights, input_layer,output_layer=actor.create_actor_network(state_size,action_size)
gradients=tf.gradients(output_layer,input_layer)
print(model.summary())

print(gradients)
