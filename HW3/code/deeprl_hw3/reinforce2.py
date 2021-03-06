import gym
import numpy as np
import keras
import time
import random
import tensorflow as tf
import copy
from keras import backend as K
from keras import optimizers
import time
def get_total_reward(env, model):
    """compute total reward

    Parameters
    ----------
    env: gym.core.Env
      The environment. 
    model: (your action model, which can be anything)

    Returns
    -------
    total_reward: float
    """
    state=env.reset()
    state=process_state(state)
    total_reward=0
    done=False
    int_cnt=0
    while done==False:
        action,softmax=choose_action(model,state)
        state,reward,done,info=env.step(action)
        state=process_state(state)
        total_reward=total_reward+reward
        int_cnt=int_cnt+1
    return total_reward,int_cnt


def choose_action(model, observation):
    """choose the action 

    Parameters
    ----------
    model: (your action model, which can be anything)
    observation: given observation
    in this problom, the model is a keras model loaded from configuration file
    the observation is the state from environment, 
    the output is the softmax activation function based probability
    

    Returns
    -------
    p: float 
        probability of action 1
    action: int
        the action you choose
    """
    softmax=model.predict(observation)
    totals = []
    running_total = 0

    for i in range(softmax.size):
        running_total += softmax[0,i]
        totals.append(running_total)
    rnd = random.random() * running_total
    for action in range(len(totals)):
        if rnd < totals[action]:
            return action,softmax


def reinforce(env):
    """Policy gradient algorithm

    Parameters
    ----------
    env: your environment

    Returns
    -------
    total_reward: float
    """
    return 0


def process_state(state):
    state_vector=np.zeros((state.shape[0],1))
    state_vector[:,0]=state
    state_T=np.transpose(state_vector)
    return state_T
def run_one_episode(env,nn):    
    states=[]
    actions=[]
    dones=[]
    softmaxes=[]
    rewards=[]
    state=env.reset()
    state=process_state(state)
    done=False
    int_cnt=0
    while done==False:
        states.append(state)
        action,softmax=choose_action(nn,state)
        actions.append(action)
        softmaxes.append(softmax)
        state,reward,done,info=env.step(action)
        rewards.append(reward)
        #dones.append(done)
        int_cnt=int_cnt+1
        state=process_state(state)
 
    return states,actions,rewards,softmaxes,dones

def train_nn(sess,nn,states,actions,rewards,softmaxes,gamma,lr):
    rewards=discount_rewards(rewards,gamma)
    length=len(states)
    output_layer=nn.outputs
    #print('output_laysers',output_layer)
    log_output=tf.log(output_layer)
    #the loss tensor of action 0
    loss0=log_output[0,:,0]
    #print(loss0)
    #the loss tensor of action 1
    loss1=log_output[0,:,1]
    #print(loss1)
    #partial derivitive with respect to action 0
    gradient_cal0=tf.gradients(loss0, nn.weights)
    gradient_cal1=tf.gradients(loss1, nn.weights)
    
    phs=[]
    ops=[]
    for i in range(len(nn.weights)): 
        phs=np.append(phs,K.placeholder(shape=nn.weights[i].get_shape()))
        ops=np.append(ops,K.update_add(nn.weights[i],phs[i]))
    
    #start_time=time.time()
    for i in range(length):
        scale=np.float_power(gamma,i)*lr*rewards[i]
        state=states[i]
        action=actions[i]
        if action==0:
           #print("using gradients0")
           gradients=sess.run(gradient_cal0,feed_dict={nn.input:state})
        else:
           #print("using gradients1")
           gradients=sess.run(gradient_cal1,feed_dict={nn.input:state})
        scaled_gradients=np.multiply(scale,gradients)        
        '''
        tensor_gradient=K.variable(value=scaled_gradients[0])  
        op=K.update(nn.weights[0],tensor_gradient)
        
        #print(K.eval(tensor_gradient))
        sess.run(op,feed_dict={tensor_gradient:scaled_gradients[0]})
        print("After update")
        print(K.eval(nn.weights[0]))
        '''        
        for ind in range(len(scaled_gradients)):
            #before=K.eval(nn.weights[ind])
            sess.run(ops[ind],feed_dict={phs[ind]:scaled_gradients[ind]})
            #after=K.eval(nn.weights[ind])
            #print(np.array_equal(before,after))
    #end_time=time.time()
    #print('time:',end_time-start_time)
    return nn


def discount_rewards(rewards,gamma):
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, len(rewards))):
        if rewards[t] != 0:
           running_add = running_add * gamma + rewards[t]
           discounted_rewards[t] = running_add
    return discounted_rewards

