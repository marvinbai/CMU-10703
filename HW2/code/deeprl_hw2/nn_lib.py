import gym
import tensorflow as tf
import numpy as np
import keras
import operator
import os

from keras.layers import Input, Dense, Convolution2D, BatchNormalization
from keras.layers import Flatten, Lambda, Conv2D
from keras.models import Model
from keras import backend as K
from keras.models import load_model
from deeprl_hw2.improcess import AtariProcessor
from deeprl_hw2.improcess import HistoryStore
from deeprl_hw2.policy import GreedyPolicy
from deeprl_hw2.policy import GreedyEpsilonPolicy


# Reference: https://github.com/matthiasplappert/keras-rl/blob/master/rl/util.py
def huber_loss(y_true, y_pred, clip_value):
    assert clip_value > 0.
    x = y_true - y_pred
    if np.isinf(clip_value):
        return .5 * K.square(x)
    condition = K.abs(x) < clip_value
    squared_loss = .5 * K.square(x)
    linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        if hasattr(tf, 'select'):
            return tf.select(condition, squared_loss, linear_loss)  # condition, true, false
        else:
            return tf.where(condition, squared_loss, linear_loss)  # condition, true, false
    elif K.backend() == 'theano':
        from theano import tensor as T
        return T.switch(condition, squared_loss, linear_loss)
    else:
        raise RuntimeError('Unknown backend "{}".'.format(K.backend()))

def clipped_error(y_true, y_pred):
    return K.mean(K.mean(huber_loss(y_true, y_pred, 1), axis = 0), axis = -1)
    # return K.mean(huber_loss(y_true, y_pred, 1), axis=-1)


def model_create_linear(input_size, output_size):
    input = Input(shape = (input_size, ), name = 'input')
    input_norm = BatchNormalization()(input)
    output = Dense(output_size, activation = 'linear')(input_norm)
    model_linear = Model(input, output)
    print(model_linear.summary())
    return model_linear

def model_create_cnn(input_size, output_size):
    input = Input(shape = (input_size[0], input_size[1], input_size[2]), name = 'input')
    input_norm = BatchNormalization()(input)
    h1 = Conv2D(32, (8, 8), strides = 4, activation = "relu")(input_norm)
    h2 = Conv2D(64, (4, 4), strides = 2, activation = "relu")(h1)
    h3 = Conv2D(64, (3, 3), strides = 1, activation = "relu")(h2)
    h3_flatten = Flatten()(h3)
    h4 = Dense(512, activation = "relu")(h3_flatten)
    output = Dense(output_size, activation = "linear")(h4)
    model_cnn = Model(input, output)
    print(model_cnn.summary())
    return model_cnn




def model_create_cnn_duel(input_size, output_size):
    input = Input(shape = (input_size[0], input_size[1], input_size[2]), name = 'input')
    input_norm = BatchNormalization()(input)
    h1 = Convolution2D(32, (8, 8), strides = 4, activation = "relu")(input_norm)
    h2 = Convolution2D(64, (4, 4), strides = 2, activation = "relu")(h1)
    h3 = Convolution2D(64, (3, 3), strides = 1, activation = "relu")(h2)
    h3_flatten = Flatten()(h3)
    h4 = Dense(512, activation = "relu")(h3_flatten)
    h5 = Dense(output_size + 1, activation = "linear")(h4)
    output = Lambda(lambda_func, output_shape = output_of_lambda)(h5)
    model_cnn_duel = Model(input, output)
    print(model_cnn_duel.summary())
    return model_cnn_duel


def output_of_lambda(input_shape):
    return (input_shape[0], input_shape[1] - 1)

def lambda_func(x):
    # x 32 x 7.
    # V 32 x 6.
    V = K.repeat_elements(x[:,:1], K.int_shape(x)[1] - 1, axis = 1)
    # A 32 x 6.
    A = x[:, 1:K.int_shape(x)[1]]
    # tmp should be 32 x 1.
    tmp = K.mean(A, axis = 1, keepdims = True)
    # Q should be 32 x 6.
    Q = V + A - K.repeat_elements(tmp, K.int_shape(x)[1] - 1, axis = 1)
    return Q



def compile(model):
    # adam = keras.optimizers.Adam(lr = 0.00025)
    rmsprop = keras.optimizers.RMSprop(lr = 0.00025, rho = 0.95, epsilon = 1e-08, decay = 0.0)
    # model.compile(optimizer = adam, loss = clipped_error)
    model.compile(optimizer = rmsprop, loss = clipped_error)
    print("Model compiled.")
    return model 

def model_predict(model, state):
    # Output should be 1x6.
    output = model.predict_on_batch(state)
    return output

def model_predict_class(model, state):
    output = model.predict_on_batch(state)
    q_index, q = max(enumerate(output[0]), key = operator.itemgetter(1))
    return q, q_index

def model_train_on_batch(model, x_train, x_target, epoch_num):
    for i in range(epoch_num):
        loss = model.train_on_batch(x_train, x_target)
        #if (i == 1) or (i == epoch_num - 1):
        #    print("At epoch", i, ", loss is", loss)	
    return model

# Evaluate neural network model for a environment (env will be changed.)
# Return total reward after taking actions according to NN.
def model_evaluate(model, env, IMAGE_SIZE, HISTORY_LENGTH):    
    # Initialize everything.
    observation = env.reset()
    atari_processor = AtariProcessor(IMAGE_SIZE)
    history_store = HistoryStore(HISTORY_LENGTH, IMAGE_SIZE)
    greedy_selector = GreedyEpsilonPolicy(0.1)
    reward_cum = 0 # Cumulative total reward.
    done = False 
    cnt_interaction = 0
    # Run and calculate cumulative reward until reaching terminate state.
    while done == False:
        state = atari_processor.state_for_nn(observation)
        history_store.add_history(state)
        nn_input = history_store.get_history_for_nn()
        q_values = model_predict(model, nn_input)
        action = greedy_selector.select_action(q_values)
        observation, reward, done, info = env.step(action)
        reward_cum += reward
        cnt_interaction += 1
    # print("Total reward is", reward_cum)
    return reward_cum, cnt_interaction

def model_evaluate_cnn(model, env, IMAGE_SIZE, HISTORY_LENGTH):    
    # Initialize everything.
    observation = env.reset()
    atari_processor = AtariProcessor(IMAGE_SIZE)
    history_store = HistoryStore(HISTORY_LENGTH, IMAGE_SIZE)
    greedy_selector = GreedyEpsilonPolicy(0.1)
    reward_cum = 0 # Cumulative total reward.
    done = False 
    cnt_interaction = 0
    # Run and calculate cumulative reward until reaching terminate state.
    while done == False:
        state = atari_processor.state_for_nn(observation)
        history_store.add_history(state)
        nn_tmp = history_store.get_history()
        nn_input = np.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], HISTORY_LENGTH), dtype=float)
        nn_input[0, :] = nn_tmp
        q_values = model_predict(model, nn_input)
        action = greedy_selector.select_action(q_values)
        observation, reward, done, info = env.step(action)
        reward_cum += reward
        cnt_interaction += 1
    # print("Total reward is", reward_cum)
    return reward_cum, cnt_interaction

# Eat an environment and take random actions, return total reward.
def random_evaluate(env):
    env.reset()
    reward_cum = 0
    done = False
    while done == False:
        # Take random action.
        observation, reward, done, info = env.step(env.action_space.sample())
        reward_cum += reward
    print("Total reward is", reward_cum)
    return reward_cum

def model_save_weight(model, number):
    path = "./weights"
    if not os.path.exists(path):
        os.makedirs(path)
    os.chdir(path)
    name = "weight" + str(number) + ".h5"
    model.save_weights(name)
    os.chdir("..")
    return None

class NN_linear: 
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.model = model_create_linear(input_size, output_size)
        self.model = compile(self.model)
    
    def predict(self, batch_input):
        # batch_input should be 32 x 28224.
        # q_values is 32 x 6.
        q_values = model_predict(self.model, batch_input)  
        return q_values

    def predict_advance(self, batch_input):
        # batch_input should be 32 x 28224.
        # q_values is 32 x 6.
        q_values = model_predict(self.model, batch_input)  
        # y is 1 x 32 numpy array.
        y = np.max(q_values, axis = 1)
        # action is 1 x 32 numpy array.
        action = np.argmax(q_values, axis = 1)
        return q_values, y, action

    def train(self, batch_input, y_target, epoch_num):
        loss = 0
        for i in range(epoch_num):
            loss += self.model.train_on_batch(batch_input, y_target)
        loss = loss / epoch_num
        return loss

class NN_cnn: 
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.model = model_create_cnn(input_size, output_size)
        self.model = compile(self.model)
    
    def predict(self, batch_input):
        # batch_input should be 32 x 84 x 84 x 4.
        # q_values is 32 x 6.
        q_values = model_predict(self.model, batch_input)  
        return q_values

    def predict_advance(self, batch_input):
        # batch_input should be 32 x  84 x 84 x 4.
        # q_values is 32 x 6.
        q_values = model_predict(self.model, batch_input)  
        # y is 1 x 32 numpy array.
        y = np.max(q_values, axis = 1)
        # action is 1 x 32 numpy array.
        action = np.argmax(q_values, axis = 1)
        return q_values, y, action

    def train(self, batch_input, y_target, epoch_num):
        loss = 0
        for i in range(epoch_num):
            loss += self.model.train_on_batch(batch_input, y_target)
        loss = loss / epoch_num
        return loss


class NN_cnn_duel: 
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.model = model_create_cnn_duel(input_size, output_size)
        self.model = compile(self.model)
    
    def predict(self, batch_input):
        # batch_input should be 32 x 84 x 84 x 4.
        # q_values is 32 x 6.
        q_values = model_predict(self.model, batch_input)  
        return q_values

    def predict_advance(self, batch_input):
        # batch_input should be 32 x  84 x 84 x 4.
        # q_values is 32 x 6.
        q_values = model_predict(self.model, batch_input)  
        # y is 1 x 32 numpy array.
        y = np.max(q_values, axis = 1)
        # action is 1 x 32 numpy array.
        action = np.argmax(q_values, axis = 1)
        return q_values, y, action

    def train(self, batch_input, y_target, epoch_num):
        loss = 0
        for i in range(epoch_num):
            loss += self.model.train_on_batch(batch_input, y_target)
        loss = loss / epoch_num
        return loss



