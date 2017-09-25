import gym
import numpy as np
#import keras
import deeprl_hw3.arm_env
from deeprl_hw3.controllers import simulate_dynamics,approximate_A,approximate_B,calc_lqr_input
import copy
import time
import os
fp1='LQR_2_q.txt'
fp2='LQR_2_dq.txt'
fp3='LQR_2_u.txt'
fp4='LQR_2_r.txt'

f1 = open(fp1, 'w')
f2 = open(fp2, 'w')
f3 = open(fp3, 'w')
f4 = open(fp4, 'w')


env = gym.make('TwoLinkArm-limited-torque-v0')
prev_state=env.reset()
print(prev_state)
actions=env.action_space
print(actions)
print(actions.shape)
observations=env.observation_space
print(observations.high)
print(observations.low)

print("Action High:",actions.low)
print("Action Low:",actions.high)
print("Goal:",env.goal)
sim_env=copy.deepcopy(env)
done=False
rewards=0
while done==False:
    env.render( mode='human')
    #time.sleep(0.5)
    q=env.position
    f1.write(str(q)+'\n')
    dq=env.velocity
    f2.write(str(dq)+'\n')
    u=calc_lqr_input(env, sim_env,mode='continous')
    print('u:',u)
    f3.write(str(u)+'\n')
    x,reward,done,info=env.step(u)
    print('x:',x)
    rewards=rewards+reward
    f4.write(str(rewards)+'\n')
    print("rewards:",rewards)
    print(done)
    print(info)
f1.close()
f2.close()
f3.close()
f4.close()
