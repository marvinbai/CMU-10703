import gym
import numpy as np
import tensorflow as tf
import deeprl_hw2.nn_lib as nn_lib

from gym import wrappers
from deeprl_hw2.improcess import AtariProcessor
from deeprl_hw2.improcess import HistoryStore
from deeprl_hw2.policy import GreedyPolicy


IMAGE_SIZE = (84,84)
HISTORY_LENGTH = 4

env = gym.make('BreakoutDeterministic-v0')
env = wrappers.Monitor(env, './video')
observation = env.reset()
num_actions = env.action_space.n

model_cnn = nn_lib.NN_cnn((IMAGE_SIZE[0], IMAGE_SIZE[1], HISTORY_LENGTH), num_actions)
model_cnn.model.load_weights('weights12700.h5')

atari_processor = AtariProcessor(IMAGE_SIZE)
history_store = HistoryStore(HISTORY_LENGTH, IMAGE_SIZE)
greedy_selector = GreedyPolicy()
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
    q_values = model_cnn.predict(nn_input)
    action = greedy_selector.select_action(q_values)
    observation, reward, done, info = env.step(action)
    reward_cum += reward
    cnt_interaction += 1

print("Total reward is", reward_cum)
print("Total interaction is", cnt_interaction)




