import numpy as np

class MSample:
    def __init__(self,state,action,reward,done,info,next_state):
        self.state=state
        self.action=action
        self.reward=reward
        self.done=done
        self.info=info
        self.next_state=next_state
class MMStore:
    def __init__(self, mem_size=1000000):
        self.mem_size=mem_size
        self.store=[None]*self.mem_size
        self.pt=0
        self.status=False
    def append(self,state,action,reward,done,info,next_state):
        tmp_msample=MSample(state,action,reward,done,info,next_state)
        if (self.pt>self.mem_size-1):
            # full,set  pointer to the oldest value
            self.pt=self.pt-self.mem_size
            self.status=True
        self.store[self.pt]=tmp_msample
        self.pt=self.pt+1
        return True
    def sample(self,batch_size,state_size,action_size):
        if(self.status==True):
            #memory is full, sample from full memory
            choice=np.random.random_integers(0,self.mem_size-1,batch_size)
        else:
            # sample from zero to pt-1 position
            choice=np.random.random_integers(0,self.pt-1,batch_size)
        states=np.zeros((batch_size,state_size))
        actions=np.zeros((batch_size,action_size))
        rewards=np.zeros((batch_size,1))
        dones=np.zeros((batch_size,1),dtype=bool)
        infos=np.zeros((batch_size,1),dtype=str)
        next_states=np.zeros((batch_size,state_size))

        for i in range(choice.size):
            tmp_msample=self.store[choice[i]]
            states[i,:]=tmp_msample.state
            actions[i,:]=tmp_msample.action
            rewards[i,:]=tmp_msample.reward
            dones[i,:]=tmp_msample.done
            infos[i,:]=tmp_msample.info
            next_states[i:,]=tmp_msample.next_state

        return states,actions,rewards, dones,infos,next_states
    def get_length(self):
        if(self.status==True):
            return self.mem_size
        else:
            return self.pt
    def ratio_fill(self,env,ratio,MAX_EPI_INT):
        fill_cnt=0
        fill_flag=False
        while(fill_flag==False):
            episode_interaction_cnt=0
            epi_flag=False
            pre_observation=env.reset()
            while(epi_flag==False):
                action=env.action_space.sample()
                observation, reward, done, info=env.step(action)
                #print(done)
                self.append(pre_observation,action,reward,done,info,observation)
                pre_observation=observation         
                episode_interaction_cnt=episode_interaction_cnt+1
                fill_cnt=fill_cnt+1
                if(episode_interaction_cnt>MAX_EPI_INT)|(done==True):
                    epi_flag=True 
            if fill_cnt>ratio*self.mem_size:
                fill_flag=True   
        return True

    def sample_dataSelection(self, batch_size, state_size, action_size, repetitionNum):
        states, actions, rewards, dones, infos, next_states = self.sample(batch_size, state_size, action_size)
        states_new = states
        actions_new = actions
        rewards_new = rewards
        dones_new = dones
        infos_new = infos
        next_states_new = np.zeros(next_states.shape)
        for i in range(repetitionNum - 1):
            states_2, actions_2, rewards_2, dones_2, infos_2, next_states_2 = self.sample(batch_size, state_size, action_size)
            states = np.append(states, states_2, axis = 0)
            actions = np.append(actions, actions_2, axis = 0)
            rewards = np.append(rewards, rewards_2, axis = 0)
            dones = np.append(dones, dones_2, axis = 0)
            infos = np.append(infos, infos_2, axis = 0)
            next_states = np.append(next_states, next_states_2, axis = 0)
        rewards_index = np.argsort(np.abs(rewards), axis = 0)[::-1]
        for i in range(batch_size):
            states_new[i] = states[rewards_index[i]]
            actions_new[i] = actions[rewards_index[i]]
            rewards_new[i] = rewards[rewards_index[i]]
            dones_new[i] = dones[rewards_index[i]]
            infos_new[i] = infos[rewards_index[i]]
            next_states_new[i] = next_states[rewards_index[i]]
        states = states_new
        actions = actions_new
        rewards = rewards_new
        dones = dones_new
        infos = infos_new
        next_states = next_states_new
            
        return states, actions, rewards, dones, infos, next_states    
    
    
    