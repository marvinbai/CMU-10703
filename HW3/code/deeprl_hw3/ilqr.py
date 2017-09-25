"""LQR, iLQR and MPC."""

from deeprl_hw3.controllers import approximate_A, approximate_B
import numpy as np
import scipy.linalg
import copy

def simulate_dynamics_next(env, x, u):
    """Step simulator to see how state changes.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.
    u: np.array
      The command to test. When approximating B you will need to
      perturb this.

    Returns
    -------    
    next_x: np.array
    """
    #delta=1e-5
    q=copy.deepcopy(env.position)
    dq=copy.deepcopy(env.velocity)
    state=copy.deepcopy(env.state)
    env.state=copy.deepcopy(x)
    next_state,reward,done,info=env._step(u)
    #reset env
    env.state=copy.deepcopy(state)
    return next_state


def cost_inter(env, x, u):
    """intermediate cost function

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.
    u: np.array
      The command to test. When approximating B you will need to
      perturb this.

    Returns
    -------
    l, l_x, l_xx, l_u, l_uu, l_ux. The first term is the loss, where the remaining terms are derivatives respect to the
    corresponding variables, ex: (1) l_x is the first order derivative d l/d x (2) l_xx is the second order derivative
    d^2 l/d x^2
    """
    delta=1e-5
    inter_scale=1e-2
    #store env
    q=copy.deepcopy(env.position)
    dq=copy.deepcopy(env.velocity)
    state=copy.deepcopy(env.state)
    
    x_dim=x.shape[0] 
    u_dim=u.shape[0]
    l=np.square(np.linalg.norm(u))
    l_x=np.zeros(x.shape)
    l_xx=np.zeros((x_dim,x_dim))
    l_u=2*u
    l_uu=np.eye(u_dim)
    l_uu=2*l_uu
    l_ux=np.zeros((u_dim,x_dim))
    #restore env
    env.state=copy.deepcopy(state)  
 
    l=l*inter_scale
    l_x=l_x*inter_scale
    l_xx=l_xx*inter_scale
    l_u=l_u*inter_scale
    l_uu=l_uu*inter_scale
    l_ux=l_ux*inter_scale
    
    return l, l_x, l_xx, l_u, l_uu, l_ux


def cost_final(env, x):
    """cost function of the last step

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.

    Returns
    -------
    l, l_x, l_xx The first term is the loss, where the remaining terms are derivatives respect to the
    corresponding variables
    """
    delta=1e-5
    final_scale=1e6
    #save env
    q=copy.deepcopy(env.position)
    dq=copy.deepcopy(env.velocity)
    state=copy.deepcopy(env.state)

    delta=1e-5
    goal=copy.deepcopy(env.goal)
    x_dim=x.shape[0]
    l=final_scale*np.square(np.linalg.norm(x-goal))
    l_x=2*final_scale*(x-goal)
    l_xx=2*final_scale*np.eye(x_dim)
    #restore env
    env.state=copy.deepcopy(state) 
    return l,l_x,l_xx


def simulate(env, x0, U):
    #save state
    state=copy.deepcopy(env.state)
    goal=copy.deepcopy(env.goal)
    #start trojectory simulation
    env.state=copy.deepcopy(x0)
    X=[]
    X.append(x0)
    rewards=[]
    inter_cost_sum=0
    inter_cost_list=[]
    accu_inter_cost_list=[]
    inter_scale=1e-2
    final_scale=1e6
    for i in range(len(U)):
        x,reward,done,info=env.step(U[i])
        X.append(x)
        rewards.append(reward)
        inter_cost=np.square(np.linalg.norm(U[i]))*inter_scale
        inter_cost_list.append(inter_cost)
        inter_cost_sum=inter_cost_sum+inter_cost
        accu_inter_cost_list.append(inter_cost_sum)
    final_cost=final_scale*np.square(np.linalg.norm(x-goal))
    cost_sum=inter_cost_sum+final_cost
    cost_list=accu_inter_cost_list+final_cost
    
    #restore env
    env.state=copy.deepcopy(state)
    #note the return of X has one more element, the final x
    return X,cost_list,inter_cost_list,accu_inter_cost_list,inter_cost_sum,final_cost,cost_sum

def simulate_AB(env,X,U):
    #store env
    state=copy.deepcopy(env.state)
    A_list=[]
    B_list=[]
    for i in range(len(U)):
        xt=X[i]
        ut=U[i]
        A=approximate_A(env,xt,ut,delta=1e-5,dt=1e-3,mode='discrete')
        B=approximate_B(env,xt,ut,delta=1e-5,dt=1e-3,mode='discrete')
        A_list.append(A)
        B_list.append(B)
    env.state=copy.deepcopy(state)
    #return the lists of A, B from finite difference
    return A_list,B_list

def calc_ilqr_input(env, sim_env, tN=50, max_iter=1e6):
    """Calculate the optimal control input for the given state.


    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use this environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.
    tN: number of control steps you are going to execute
    max_itr: max iterations for optmization

    Returns
    -------
    U: np.array
      The SEQUENCE of commands to execute. The size should be (tN, #parameters)
    """
    threshold=1e-7
    lam=1
    lam_scale=10
    x0=copy.deepcopy(env.state)
    #random initialize control list
    U_list=[]
    for i in range(tN):
        u=np.array(np.random.randn(2,))
        #print(u)
        #print(u.shape)
        U_list.append(u)    
    #do a forward pass with U to get X
    X_list,cost_list,inter_cost_list,accu_inter_cost_list,inter_cost_sum,final_cost,cost_sum=simulate(env,x0,U_list)
    prev_cost_sum=copy.deepcopy(cost_sum)    

    #start training with random sequence
    train_flag=True
    train_cnt=0
    f = open('cost.txt','w')
    while train_flag==True:
        #X,cost_list,inter_cost_list,accu_inter_cost_list,inter_cost_sum,final_cost,cost_sum=simlate(env,x0,U)
        new_X_list=[]
        new_U_list=[]
        #simulate environment to get A, B matrix x_t+1=A*x_t+B*u_t
        A_list,B_list=simulate_AB(env,X_list[0:tN],U_list)
        #print('after simulate AB:',env.state)
        # do a backward pass on the control list
        k_list=[]
        K_list=[]
        #compute final state cost differential
        fl,fl_x,fl_xx=cost_final(env, X_list[tN])
        #print('after_final_cost:',env.state)
        #set backward start point
        V=fl
        V_x=fl_x
        V_xx=fl_xx
        #work backwards to solve for V, Q, k, and K for each time step
        for i in reversed(range(tN)):
            xt=X_list[i]
            ut=U_list[i]
            #inter cost differential is different for every state
            l, l_x, l_xx, l_u, l_uu, l_ux=cost_inter(env, xt, ut)
            Q_x = l_x + np.dot(A_list[i].T, V_x) 
            Q_u = l_u + np.dot(B_list[i].T, V_x)
            Q_xx = l_xx + np.dot(A_list[i].T, np.dot(V_xx, A_list[i])) 
            Q_ux = l_ux + np.dot(B_list[i].T, np.dot(V_xx, A_list[i]))
            Q_uu = l_uu + np.dot(B_list[i].T, np.dot(V_xx, B_list[i]))
            Q_uu_inv=np.linalg.inv(Q_uu)
            kt=-np.dot(Q_uu_inv, Q_u)
            Kt=-np.dot(Q_uu_inv, Q_ux)
            k_list.append(kt)
            K_list.append(Kt)
            #update V
            #delta_V =-0.5*np.dot(kt.T, np.dot(Q_uu, kt))
            V_x = Q_x - np.dot(Kt.T, np.dot(Q_uu, kt))
            V_xx = Q_xx - np.dot(Kt.T, np.dot(Q_uu, Kt))
        #print('after_back reverse:',env.state)
        #do a forward pass to get new ut and xt
        new_X_list.append(x0)
        #print('after append',new_X_list[0])
        new_x=copy.deepcopy(x0)
        for i in range(tN):
            #print('U[i]:',U[i])
            new_u=U_list[i] + 0.1*k_list[tN-i-1] + np.dot(K_list[tN-i-1], new_x-X_list[i])
            #print('u_new:',u_new)
            new_U_list.append(new_u)
            #simulate in env to get next step new_x
            new_x=simulate_dynamics_next(env, new_x, new_u)
            new_X_list.append(new_x)

        cal_new_X_list,cost_list,inter_cost_list,accu_inter_cost_list,inter_cost_sum,final_cost,cost_sum=simulate(env,x0,new_U_list)
        #print(con_new_X_list)
        #if train_cnt%10==0:
            #con_new_X_list,cost_list,inter_cost_list,accu_inter_cost_list,inter_cost_sum,final_cost,cost_sum=simulate(env,x0,new_U_list)
        print(train_cnt)
        print(cost_sum)
        f.write(str(cost_sum) + '\n')
        #update X_list U_list
        U_list=copy.deepcopy(new_U_list)
        X_list=copy.deepcopy(new_X_list)
        train_cnt=train_cnt+1        
        print('train_cnt:',train_cnt)
        #check terminal state
        if abs(prev_cost_sum-cost_sum)<threshold:
            print('Train threshold reached !!!!!!!!!!!!!!!!!!!!!!!!!!!')
            train_flag=False
        prev_cost_sum=cost_sum
        if train_cnt>=max_iter:
            train_flag=False  
    f.close()
    return U_list
    

