import numpy as np
import math
#from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential, Model
#from keras.engine.training import collect_trainable_weights
from keras.layers import Dense, Flatten, Input, merge, Lambda, BatchNormalization
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
from keras import initializers
from keras import optimizers

H1_UNITS = 400
H2_UNITS = 300

class ActorNetwork:
    def __init__(self, sess, state_size, action_size, batch_size, soft_update, lr):
        self.sess = sess
        self.batch_size = batch_size
        self.soft_update = soft_update
        self.lr = lr

        K.set_session(sess)

        #Now create the model
        self.online_nn , self.online_weights, self.online_state ,self.online_output= self.create_actor_network(state_size, action_size)   
        self.target_nn, self.target_weights, self.target_state,self.target_output = self.create_actor_network(state_size, action_size) 
        
        self.action_grads = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.online_output, self.online_weights, -self.action_grads)

        self.grads = zip(self.params_grad, self.online_weights)

        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(self.grads)

        self.sess.run(tf.global_variables_initializer())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.online_state: states,
            self.action_grads: action_grads
        })

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

    def create_actor_network(self, state_size,action_size):
        #initializer1=initializers.RandomUniform(minval=-1/np.sqrt(state_size), maxval=1/np.sqrt(state_size), seed=None)
        #initializer2=initializers.RandomUniform(minval=-1/np.sqrt(H1_UNITS), maxval=1/np.sqrt(H1_UNITS), seed=None)
        #initializer3=initializers.RandomUniform(minval=-0.003, maxval=0.003, seed=None)

        #prepare actor layers
        input_layer = Input(shape=(state_size, ))
        #input_layer_norm=BatchNormalization()(input_layer)
        h1= Dense(H1_UNITS, activation = 'relu')(input_layer)
        #h1= Dense(H1_UNITS, activation = 'relu',kernel_initializer=initializer1,bias_initializer=initializer1)(input_layer)
        h2= Dense(H2_UNITS, activation = 'relu')(h1)
        #h2= Dense(H2_UNITS, activation = 'relu',kernel_initializer=initializer2,bias_initializer=initializer2)(h1)
        output_layer= Dense(action_size, activation = 'tanh')(h2)
        #output_layer= Dense(action_size, activation = 'tanh',kernel_initializer=initializer3,bias_initializer=initializer3)(h2)
        model= Model(input_layer, output_layer)
        #adam = Adam(lr=self.lr)
        #model.compile(loss='mse', optimizer=adam)
 
        return model, model.trainable_weights, input_layer,output_layer


    def compare(self):
        w1=self.online_nn.get_weights()
        w2=self.target_nn.get_weights()
        for i in range(len(w1)):
            tmp=np.array_equal(w1[i],w2[i])
            if tmp==False:
                return False
        return True

