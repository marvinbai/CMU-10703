import numpy as np
import math
#from keras.initializations import normal, identity
from keras.models import model_from_json, load_model
#from keras.engine.training import collect_trainable_weights
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation , BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
from keras.layers.merge import add
from keras import initializers
from keras import optimizers

H1_UNITS = 400
H2_UNITS = 300

class CriticNetwork:
    def __init__(self, sess, state_size, action_size, batch_size, soft_update, lr):
        self.sess = sess
        self.batch_size = batch_size
        self.soft_update = soft_update
        self.lr = lr
        self.action_size = action_size
        self.state_size=state_size
        K.set_session(sess)

        #Now create the model
        self.online_nn, self.online_state, self.online_action,self.online_output = self.create_critic_network(state_size, action_size)  
        self.target_nn, self.target_state, self.target_action,self.target_output = self.create_critic_network(state_size, action_size)

        self.action_grads = tf.gradients(self.online_output, self.online_action)
        self.sess.run(tf.global_variables_initializer())
        #self.target_nn.set_weights(self.online_nn.get_weights())
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

    def action_gradients(self,states,actions):
        #print(self.online_output)
        #print(self.online_state)
        #print(self.online_action)
        self.action_grads = tf.gradients(self.online_output, self.online_action)

        #print(self.action_grads)
        grads= self.sess.run(self.action_grads, feed_dict={self.online_state:states, self.online_action:actions})[0]
        #print(grads)
        return grads
    def replace_none_with_zero(self,l):
        return [0 if i==None else i for i in l] 
    def copy(self):
        self.target_nn.set_weights(self.online_nn.get_weights())
    def soft_target_nn_update(self):
        online_weights = self.online_nn.get_weights()
        target_weights = self.target_nn.get_weights()
        c1=self.soft_update
        c2=1-self.soft_update

        for i in range(len(target_weights)):
            target_weights[i]=c2*target_weights[i]+c1*online_weights[i]
        self.target_nn.set_weights(target_weights)


    def create_critic_network(self, state_size,action_size):
        #build crtic network
        #initializer1=initializers.RandomUniform(minval=-1/np.sqrt(state_size), maxval=1/np.sqrt(state_size), seed=None)
        #initializer2=initializers.RandomUniform(minval=-1/np.sqrt(H1_UNITS), maxval=1/np.sqrt(H1_UNITS), seed=None)
        #initializer0=initializers.RandomUniform(minval=-1/np.sqrt(action_size), maxval=1/np.sqrt(action_size), seed=None)
        #initializer3=initializers.RandomUniform(minval=-0.003, maxval=0.003, seed=None)

        S = Input(shape=(state_size,))
        A = Input(shape=(action_size,))
        #S1 = Dense(H1_UNITS, activation='relu',kernel_initializer=initializer1,bias_initializer=initializer1)(S)
        S1 = Dense(H1_UNITS, activation='relu')(S)

        A1 = Dense(H2_UNITS, activation='linear')(A) 
        S2 = Dense(H2_UNITS, activation='linear')(S1)
        M = add([S2,A1])    
        H = Dense(H2_UNITS, activation='relu')(M)
        #V = Dense(1,activation='linear',kernel_initializer=initializer3,bias_initializer=initializer3)(H)
        V = Dense(1,activation='linear')(H)

        model= Model(inputs=[S,A],outputs=V)
        adam = Adam(lr=self.lr)
        model.compile(loss='mse', optimizer=adam)
        return model,S,A,V
    def compare(self):
        w1=self.online_nn.get_weights() 
        w2=self.target_nn.get_weights()
        for i in range(len(w1)):
            tmp=np.array_equal(w1[i],w2[i])
            if tmp==False:
                return False
        return True
    
