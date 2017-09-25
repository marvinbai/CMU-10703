import gym
import numpy as np
import scipy as sp
import os
import time
import deeprl_hw3.arm_env
import copy
from deeprl_hw3.ilqr import simulate,cost_inter,cost_final,simulate_AB,calc_ilqr_input,simulate_dynamics_next
N=100
MAX_ITER=10000
env = gym.make('TwoLinkArm-v0')
x0=env.reset()
#print('x0:',x0)
actions=env.action_space
#print(actions)
#print(actions.shape)
observations=env.observation_space
#print(observations.high)
#print(observations.low)
goal=env.goal
#print('goal:',goal)
f_q = open('q_and_dq.txt','w')
f_u = open('u.txt','w')
U=[]

for i in range(N):
    u=np.random.randn(2,)
    U.append(u)
#print('U:',len(U))
#print(U)
sim_env=copy.deepcopy(env)

X,cost_list,inter_cost_list,accu_inter_cost_list,inter_cost_sum,final_cost,cost_sum=simulate(env,x0,U)

x=copy.deepcopy(x0)
X_step_by=[]
X_step_by.append(x)
for i in range(N):
    u=U[i]
    x=simulate_dynamics_next(env,x,u)
    X_step_by.append(x)

#print('X:',len(X))
#print(X)
#print('X_step_by:',len(X_step_by))
#print(X_step_by)

'''
print(len(cost_list))
print(cost_list)
print(len(inter_cost_list))
print(inter_cost_list)
print(len(accu_inter_cost_list))
print(accu_inter_cost_list)
print(inter_cost_sum)
print(final_cost)
print(cost_sum)
print(X[1].shape)
print(U[1].shape)
'''
l, l_x, l_xx, l_u, l_uu, l_ux=cost_inter(env, X[1], U[1])
X_sub=X[0:100]
'''
print(len(X_sub))
print(X[100])

print('check intermediate cost....................')
print('l:',l)
print(l_x.shape)
print(l_x)
print(l_xx.shape)
print(l_xx)
print(l_u.shape)
print(l_u)
print(l_uu.shape)
print(l_uu)
print(l_ux.shape)
print(l_ux)

print('check final cost.....................')
'''
fl,flx,flxx=cost_final(env, X[100])
'''
print(fl)
print(flx.shape)
print(flx)
print(flxx.shape)
print(flxx)
'''

optimal_U=calc_ilqr_input(env, sim_env, tN=100, max_iter=1000)

X,cost_list,inter_cost_list,accu_inter_cost_list,inter_cost_sum,final_cost,cost_sum=simulate(env,x0,optimal_U)

for i in range(len(optimal_U)):
    # print(i)
    # env.render()
    time.sleep(0.2)
    x,reward, done, info= env._step(optimal_U[i])
    # print(x)
    # print(done)
    # f.write(str(optimal_U[i])+';'+str(x)+';'+str(reward)+';'+str(done)+'\n')
    f_q.write(str(x) + '\n')
    f_u.write(str(optimal_U[i]) + '\n')
f_q.close()
f_u.close()
print('Goal q:', env.goal_q)
print('Goal dq:', env.goal_dq)

