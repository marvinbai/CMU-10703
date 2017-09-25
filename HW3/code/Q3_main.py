import gym
import keras
import numpy as np
import scipy as sp
import tensorflow as tf
import os
import time
from deeprl_hw3.imitation import load_model
from deeprl_hw3.reinforce2 import run_one_episode,train_nn,get_total_reward
from keras.optimizers import Adam
from keras import backend as K

MAX_TRAIN_EPOCHS=10000
EVA_INTERVAL=10
gamma=0.99
LR=0.001
STEP_SIZE=0.001
env=gym.make('CartPole-v0')
EVAL_EPISODES=100
sess = K.get_session()
nn=load_model('CartPole-v0_config.yaml',None)
nn.compile('SGD','mse',metrics=['accuracy'])
sess.run(tf.global_variables_initializer())
file_path='Q3.txt'
f = open(file_path, 'w')


train_cnt=0
eval_cnt=0
end_cnt=0
train_flag=True
while train_flag==True:
    train_start=time.time()
    states, actions, rewards,softmaxes,dones=run_one_episode(env,nn)
    
    #print(len(states))
    '''
    print(actions.shape)
    print(rewards.shape)
    print(probs.shape)
    print(softmaxes.shape)
    print(states)
    print(actions)
    print(rewards)
    print(probs)
    print(softmaxes)
    '''
    train_nn(sess,nn,states, actions, rewards,softmaxes,gamma,LR)
    train_end=time.time()
    print('Finish Train:{} with time{}'.format( train_cnt,train_end-train_start))
    #f.write('Finish Train:'+str(train_cnt)+' with time:'+str(train_end-train_start)+'\n')
    train_cnt=train_cnt+1
    if train_cnt%EVA_INTERVAL==0:
        eval_start=time.time()
        total_rewards=np.zeros((EVAL_EPISODES,1))
        interactions=np.zeros((EVAL_EPISODES,1))
        for i in range(EVAL_EPISODES):
            total_reward,interaction=get_total_reward(env,nn)
            #print(total_reward)
            total_rewards[i,0]=total_reward
            interactions[i,0]=interaction
        eval_end=time.time()
        f.write(str(np.mean(total_rewards)) + ' ' + str(np.max(total_rewards)) + ' ' + str(np.min(total_rewards)) + '\n')
        print("EVAL TIME:",eval_end-eval_start)
        print('The evaluate number is:', eval_cnt)
        #f.write('The evaluate number is:'+ str(eval_cnt)+'\n')
        print('Average total reward: {} (std: {})'.format(np.mean(total_rewards), np.std(total_rewards)))
        #f.write('AVG:'+str(np.mean(total_rewards))+'Std:'+str(np.std(total_rewards))+'\n')
        print('MAX total reward: {} ,MIN total reward {}'.format(np.max(total_rewards), np.min(total_rewards)))
        #f.write('MAX:'+str(np.max(total_rewards))+'MIN:'+str(np.min(total_rewards))+'\n')
        print('Average interactions: {} (std: {})'.format(np.mean(interactions), np.std(interactions)))
        print('MAX interaction: {} ,MIN interaction: {}'.format(np.max(interactions), np.min(interactions)))
        eval_cnt=eval_cnt+1
        nn.save_weights('Q3_'+str(train_cnt)+'.h5')
        if np.min(total_rewards)==200:
            train_flag=False
    if train_cnt>MAX_TRAIN_EPOCHS:
        train_flag=False
sess.close()
f.close()
