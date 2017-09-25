"""LQR, iLQR and MPC."""

import numpy as np
import scipy as sp
import copy
from scipy import linalg as la
def simulate_dynamics(env, x, u, dt=1e-5,mode='continous'):
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
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    x[1]: for finite mode
    
    xdot: np.array
      This is the **CHANGE** in x. i.e. (x[1] - x[0]) / dt
      If you return x you will need to solve a different equation in
      your LQR controller.
    """
    #record previous state
    q=copy.deepcopy(env.position)
    dq=copy.deepcopy(env.velocity)
    state=copy.deepcopy((env.state))

   
    if mode=='continous':

        env.state=copy.deepcopy(x)
        next_state,reward, done,info=env._step(u,dt)
        env.state=copy.deepcopy(state)
        return (next_state-x)/dt
    else:
        env.state=copy.deepcopy(x)
        next_state,reward, done,info=env._step(u)
        env.state=copy.deepcopy(state)
        return next_state


def approximate_A(env, x, u, delta=1e-5, dt=1e-5,mode='continous'):
    """Approximate A matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. You will need to perturb this.
    u: np.array
      The command to test.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    A: np.array
      The A matrix for the dynamics at state x and command u.
    """
    if mode=='continous':
        A=np.zeros((x.shape[0],x.shape[0]))
        for i in range(x.shape[0]):
            xp=copy.deepcopy(x)
            xn=copy.deepcopy(x)
            xp[i]=xp[i]+delta
            xn[i]=xn[i]-delta
            xdotp=simulate_dynamics(env,xp,u,dt,'continous')
            xdotn=simulate_dynamics(env,xn,u,dt,'continous')
            d_xdot=xdotp-xdotn
            dx=2*delta
            A[:,i]=d_xdot/dx
        return A
    else:
        A=np.zeros((x.shape[0],x.shape[0]))
        for i in range(x.shape[0]):
            xp=copy.deepcopy(x)
            xn=copy.deepcopy(x)
            xp[i]=xp[i]+delta
            xn[i]=xn[i]-delta
            x1p=simulate_dynamics(env,xp,u,dt,'discrete')
            x1n=simulate_dynamics(env,xn,u,dt,'discrete')
            d_x1=x1p-x1n
            dx=2*delta            
            A[:,i]=d_x1/dx
        return A


def approximate_B(env, x, u, delta=1e-5, dt=1e-5,mode='continous'):
    """Approximate B matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test.
    u: np.array
      The command to test. You will ned to perturb this.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    B: np.array
      The B matrix for the dynamics at state x and command u.
    """
    if mode=='continous':
        B=np.zeros((x.shape[0],u.shape[0]))
        for i in range(u.shape[0]):
            up=copy.deepcopy(u)
            un=copy.deepcopy(u)
            up[i]=up[i]+delta
            un[i]=un[i]-delta
            xdotp=simulate_dynamics(env,x,up,dt,'continous')
            xdotn=simulate_dynamics(env,x,un,dt,'continous')
            d_xdot=xdotp-xdotn
            du=2*delta
            B[:,i]=d_xdot/du
        return B

    else:
        B=np.zeros((x.shape[0],u.shape[0]))
        for i in range(u.shape[0]):
            up=copy.deepcopy(u)
            un=copy.deepcopy(u)
            up[i]=up[i]+delta
            un[i]=un[i]-delta
            x1p=simulate_dynamics(env,x,up,dt,'discrete')
            x1n=simulate_dynamics(env,x,un,dt,'discrete')
            d_x1=x1p-x1n
            du=2*delta            
            B[:,i]=d_x1/du
        return B


def calc_lqr_input(env, sim_env,mode='continous'):
    """Calculate the optimal control input for the given state.

    If you are following the API and simulate dynamics is returning
    xdot, then you should use the scipy.linalg.solve_continuous_are
    function to solve the Ricatti equations.

    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use this environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.

    Returns
    -------
    u: np.array
      The command to execute at this point.
    """
    # get the current state from the true env
    state=copy.deepcopy(env.state)
    goal=copy.deepcopy(env.goal)
    Q=copy.deepcopy(env.Q)
    R=copy.deepcopy(env.R)
    #sim_u=np.random.random(2)
    sim_u=np.array([0.0,0.0])  
    A=approximate_A(sim_env, state, sim_u, delta=1e-5, dt=1e-5,mode='continous')
    #print('A:',A)
    B=approximate_B(sim_env, state, sim_u, delta=1e-5, dt=1e-5,mode='continous') 
    #print('B:',B)
    # do simulation to get A and B
    
    P=la.solve_continuous_are(A, B, Q, R)
    BT_P=np.matmul(np.matrix.transpose(B),P)
    K=np.matmul(np.linalg.inv(R),BT_P)
    u=-np.matmul(K,state-goal)    
    return u
