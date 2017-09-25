import gym
import tensorflow as tf
import numpy as np
import scipy as sp
import time
import copy
import deeprl_hw2.nn_lib as nn_lib

from gym import wrappers
from PIL import Image
from deeprl_hw2.improcess import AtariProcessor
from deeprl_hw2.improcess import HistoryStore
#from deeprl_hw2.sample import Sample
from deeprl_hw2.policy import UniformRandomPolicy
from deeprl_hw2.policy import GreedyPolicy
from deeprl_hw2.policy import GreedyEpsilonPolicy
from deeprl_hw2.policy import LinearDecayGreedyEpsilonPolicy
from deeprl_hw2.memhelpers import NNMemStore
from deeprl_hw2.nn_lib import NN_linear
from deeprl_hw2.nn_lib import NN_cnn
from keras.models import Model

# Define initial parameters.
MAX_INTERACTION = 5000000
MAX_EPISODE_LENGTH = 1000
IMAGE_SIZE = (84,84)
HISTORY_LENGTH = 4
MEM_SIZE = 800000
# MEM_SIZE = 100
BATCH_SIZE = 32
gamma = 0.99
epsilon = 0.05
C = 10000

env = gym.make("Breakout-v0")

#number of actions, used to construct an policy selector
num_actions = env.action_space.n
    
#create helpers
#observation processor
atari_processor = AtariProcessor(IMAGE_SIZE)
history_store = HistoryStore(HISTORY_LENGTH, IMAGE_SIZE)
#policy selector, for testing, use uniform random policy selector, just pass number of actions to the constructor
random_selector = UniformRandomPolicy(num_actions)
greedy_selector = GreedyPolicy()
greedy_epsilon_selector = GreedyEpsilonPolicy(epsilon)
greedy_epsilon_linear_decay_selector = LinearDecayGreedyEpsilonPolicy(1, 0.05, int(round(MAX_INTERACTION / 5, 0)))

# Initialize neural network
# Online network which changes during training but not to calculate Q*.
model_online = NN_cnn((IMAGE_SIZE[0], IMAGE_SIZE[1], HISTORY_LENGTH), num_actions)
# Fixed network which is not changed during training but to calculate Q*.
model_fixed = NN_cnn((IMAGE_SIZE[0], IMAGE_SIZE[1], HISTORY_LENGTH), num_actions)
model_fixed.model.set_weights(model_online.model.get_weights())
#model_fixed.model = Model.from_config(model_online.model.get_config())

# Initialize memory.
mem = NNMemStore(MEM_SIZE, (IMAGE_SIZE[0], IMAGE_SIZE[1], HISTORY_LENGTH))
mem.fill_half(env, random_selector, atari_processor, history_store, "matrix")

train_end = False
start = time.time()
episode_num_cnt = 0
total_interaction_cnt = 0

# Do things.
while(train_end==False):
    #prepare for new episode
    print('initialze and start new episode',episode_num_cnt)
    episode_end = False
    observation = env.reset()
    history_store.reset()
    state = atari_processor.state_for_nn(observation)
    history_store.add_history(state)
    nn_tmp = history_store.get_history()
    nn_input = np.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], HISTORY_LENGTH), dtype=float)
    nn_input[0, :] = nn_tmp

    episode_interaction_cnt = 0
    flag_first = 0
    done = False
    while done == False: 
        # Interact with environment and store into memory.

        
        q_values = model_online.predict(nn_input)
        action = greedy_epsilon_linear_decay_selector.select_action(q_values)
        observation, reward, done, info = env.step(action)
        if flag_first == 0:
            info_prev = info
            episode_end == False
            flag_first = 1
        else:
            if info != info_prev:
                episode_end = True
            else:
                episode_end = False
            info_prev = info

        reward = atari_processor.process_reward(reward)
        state_next = atari_processor.state_for_nn(observation)
        history_store.add_history(state_next)
        nn_tmp = history_store.get_history()
        nn_input_next = np.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], HISTORY_LENGTH), dtype=float)
        nn_input_next[0, :] = nn_tmp
        mem.append(nn_input, action, reward, nn_input_next, done, episode_end)
        nn_input = copy.deepcopy(nn_input_next)

        # Train neural network.
        if episode_interaction_cnt % 4 == 3:
            batch_nninput, batch_action, batch_reward, batch_next_nninput, batch_terminal, batch_life, choice = mem.sample(BATCH_SIZE, None)
            
            
            
            q_values, y, action = model_fixed.predict_advance(batch_next_nninput)
            y_target = np.zeros(q_values.shape)
            for i in range(BATCH_SIZE):
                if batch_life[i][0] == False:
                    y_target[i][batch_action[i][0].astype('uint8')] = y[i] * gamma + batch_reward[i][0]
                else:
                    y_target[i][batch_action[i][0].astype('uint8')] = batch_reward[i][0]
            
            
            
            loss = model_online.train(batch_nninput, y_target, 1)

        # Update the fixed network after C interactions.
        if total_interaction_cnt % C == 0:
            model_fixed.model.set_weights(model_online.model.get_weights())
            # model_fixed.model = Model.from_config(model_online.model.get_config())
            print("Model update.")

        # Update counter        
        episode_interaction_cnt = episode_interaction_cnt + 1    
        total_interaction_cnt = total_interaction_cnt + 1
        if(total_interaction_cnt >= MAX_INTERACTION):
            train_end = True
        if episode_interaction_cnt >= MAX_EPISODE_LENGTH:
            episode_end = True
        
    print('Finish episode:', episode_num_cnt)
    print('Finish interaction:', total_interaction_cnt)
    episode_num_cnt = episode_num_cnt + 1
    
    # Evaluate model after each 100 epoch.
    if episode_num_cnt % 100 == 0:
        reward_eval = 0
        cnt_interaction = 0
        for i in range(10):
            reward_eval_tmp, cnt_interaction_tmp = nn_lib.model_evaluate_cnn(model_online.model, env, IMAGE_SIZE, HISTORY_LENGTH)
            reward_eval += reward_eval_tmp
            cnt_interaction += cnt_interaction_tmp
        reward_eval = reward_eval / 10
        cnt_interaction = cnt_interaction / 10
        file = open("reward.txt","a")
        file.write(str(reward_eval))
        file.write("\t")
        file.write(str(cnt_interaction))
        file.write("\t")
        file.write(str(loss))
        file.write("\n")
        nn_lib.model_save_weight(model_online.model, episode_num_cnt)

file.close()
end = time.time()
print('Total time is:', end - start)
model_online.model.save_weights('weights.h5')


