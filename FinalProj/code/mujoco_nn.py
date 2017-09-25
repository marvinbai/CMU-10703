import gym
import tensorflow as tf
import numpy as np
import keras
import operator
import os

from keras.models import Sequential
from keras.layers import Input, Dense, Convolution2D, BatchNormalization,merge
from keras.layers import Flatten, Lambda
from keras.models import Model
from keras.layers.merge import add
from keras import backend as K
from keras.models import load_model
from keras import initializers
from keras import optimizers
ACTOR_LEARNING_RATE=0.0001
CRITIC_LEARNING_RATE=0.001
UPDATE_RARIO=0.001
gama=0.99
class MujocoNN:
    def __init__(self,input_shape,output_shape):
        self.learn_cnt=0
        self.input_shape=input_shape
        self.output_shape=output_shape

    def initialize(self):
        #build actor network
        actor_input_size=self.input_shape[0]
        actor_output_size=self.output_shape[0]
        initializer1=initializers.RandomUniform(minval=-1/np.sqrt(actor_input_size), maxval=1/np.sqrt(actor_input_size), seed=None)
        initializer2=initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
        initializer3=initializers.RandomUniform(minval=-0.003, maxval=0.03, seed=None)


        #prepare actor network
        actor_input_layer = Input(shape = (actor_input_size, ))
        actor_input_norm = BatchNormalization()(actor_input_layer)

        actor_h1= Dense(400, activation = 'relu',kernel_initializer=initializer1,bias_initializer=initializer1)(actor_input_norm)
        #actor_h1_norm = BatchNormalization()(actor_h1)

        actor_h2= Dense(300, activation = 'relu',kernel_initializer=initializer2,bias_initializer=initializer2)(actor_h1)
        #actor_h2_norm = BatchNormalization()(actor_h2)

        actor_output_layer= Dense(actor_output_size, activation = 'tanh',kernel_initializer=initializer3,bias_initializer=initializer3)(actor_h2)

        self.online_actor_nn= Model(actor_input_layer, actor_output_layer)
        self.target_actor_nn=Model(actor_input_layer, actor_output_layer)

        actor_adam = optimizers.Adam(lr=ACTOR_LEARNING_RATE)

        self.online_actor_nn.compile(optimizer = actor_adam, loss ="mse")
        self.online_actor_nn.compile(optimizer = actor_adam, loss ="mse")

        self.target_actor_nn.set_weights(self.online_actor_nn.get_weights())
        print(self.online_actor_nn.summary())
        print(self.target_actor_nn.summary())




        #build crtic network
        
        S = Input(shape=[actor_input_size])
        A = Input(shape=[actor_output_size]) 

        S_norm=BatchNormalization()(S)
        A_norm=BatchNormalization()(A)
        S1 = Dense(400, activation='relu')(S_norm)
        #S1_norm=BatchNormalization()(S1)

        S2 = Dense(300, activation='linear')(S1)
        #S2_norm=BatchNormalization()(S2)
        A2 = Dense(300, activation='linear')(A_norm)
        #A2_norm=BatchNormalization()(A2)

        SA = add([S2,A2])    
        H = Dense(300, activation='relu')(SA)

        #H_norm=BatchNormalization()(H)

        V = Dense(actor_output_size,activation='linear')(SA)  
        self.critic_nn = Model([S,A])

        critic_adam = optimizers.Adam(lr=CRITIC_LEARNING_RATE)
        self.critic_nn.compile(loss='mse', optimizer=critic_adam)
        return self.online_actor_nn,self.target_actor_nn
        
    def train_online_nn(self):
        return 0
    def update_target_nn(self):
        return 0
    def predict_target_nn(self):
        return 0
    def predict_online_nn(self):
        return 0
   
