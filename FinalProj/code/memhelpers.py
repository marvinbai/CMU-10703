import numpy as np



class NNSample:
    def __init__(self,nn_input,action,reward,next_nn_input,done,life):
        #nn_input is a vector
        self.nn_input=nn_input.astype('uint8')
        self.action=action
        self.reward=reward
        self.next_nn_input=next_nn_input.astype('uint8')
        self.done=done
        self.life=life

class NNMemStore:
    #default state inside the mem is for vectorized sample
    def __init__(self, mem_size=1000000, state_shape=84*84*4):
        self.mem_size=mem_size
        self.store=[None]*self.mem_size
        #the pointer of current availible position 
        self.pt=0
        self.status=False
        self.state_shape=state_shape
    def get_status(self):
        return self.status
    def find_episode_end_position(self):
        output=np.empty(0,dtype=int)
        length=self.get_mem_length()
        for i in range(length):
            tmp_nn_sample=self.store[i]
            if tmp_nn_sample.done==True:
                output=np.append(output,i)
        return output
           
    def find_life_lost_position(self):
        output=np.empty(0,dtype=int)
        length=self.get_mem_length()
        for i in range(length):
            tmp_nn_sample=self.store[i]
            if tmp_nn_sample.life==True:
                output=np.append(output,i)
        return output

    
    def get_mem_length(self):
        if self.status==False:
            return self.pt
        else:
            return self.mem_size
    def find_reward_position(self):
        output=np.empty(0,dtype=int)
        length=self.get_mem_length()
        for i in range(length):
            tmp_nn_sample=self.store[i]
            if tmp_nn_sample.reward!=0:
                output=np.append(output,i)
        return output
    def get_life_lost_sample(self):
        dt = np.dtype(NNSample)
        output=np.empty(0,dtype=dt)
        targets=self.find_life_lost_position()
        return output
    def get_reward_sample(self):
        targets=self.find_reward_position()
        return targets
    def clear(self):
        #reset pointer
        self.pointer=0
        self.store=[None]*self.mem_size
        self.status=False
        return True
    def append(self, nn_input, action, reward, next_nn_input,done,life):
        new_nn_sample=NNSample(nn_input,action,reward, next_nn_input,done,life)
        #check if the memory store is full
        if (self.pt>self.mem_size-1):
            # full,set  pointer to the oldest value
            self.pt=self.pt-self.mem_size
            self.status=True
        self.store[self.pt]=new_nn_sample
        self.pt=self.pt+1
        return True
    def get_last_sample(self):
        tmp_nn_sample=self.store[self.pt-1]
        nn_input=tmp_nn_sample.nn_input
        action=tmp_nn_sample.action
        reward=tmp_nn_sample.reward
        next_nn_input=tmp_nn_sample.next_nn_input
        done=tmp_nn_sample.done
        life=tmp_nn_sample.life
        return nn_input,action,reward, next_nn_input,done,life
    def sample(self, batch_size, indexes=None):
        batch_nn_input=np.zeros(tuple(np.append(batch_size,self.state_shape)))
        batch_reward=np.zeros((batch_size,1))
        batch_done=np.zeros((batch_size, 1), dtype=bool)
        batch_next_nn_input=np.zeros(tuple(np.append(batch_size,self.state_shape)))
        batch_action=np.zeros((batch_size,1))
        batch_life=np.zeros((batch_size,1),dtype=bool)
        if(self.status==True):
            #memory is full, sample from full memory
            choice=np.random.random_integers(0,self.mem_size-1,batch_size)
        else:
            # sample from zero to pt-1 position
            choice=np.random.random_integers(0,self.pt-1,batch_size)
        for i in range(choice.size):
            tmp_nn_sample=self.store[choice[i]]
            batch_nn_input[i,:]=tmp_nn_sample.nn_input
            batch_next_nn_input[i,:]=tmp_nn_sample.next_nn_input
            batch_reward[i,:]=tmp_nn_sample.reward
            batch_action[i,:]=tmp_nn_sample.action
            batch_done[i,:]=tmp_nn_sample.done
            batch_life[i,:]=tmp_nn_sample.life
        return batch_nn_input.astype('float32'),batch_action,batch_reward,batch_next_nn_input.astype('float32'),batch_done,batch_life,choice

    def fill_half(self,env,random_selector,atari_processor,history_store,store_type='vector',ratio=0.1):
        if(store_type=='vector'):
            print('Fill half with vector for linear NN')
        else:
            print('Fill half with matrix for CNN')
        fill_cnt=0
        fill_flag=False
        while(fill_flag==False):
            observation=env.reset()
            history_store.reset()
            state=atari_processor.state_for_nn(observation)
            life=False
            history_store.add_history(state)
            reset_flag=False
            first_step=True
            while(reset_flag==False):
                if(store_type=='vector'):
                    #print('vector')
                    nn_input=history_store.get_history_for_nn()
                else:
                    nn_input=history_store.get_history()
                action=random_selector.select_action()
                observation, reward, done, info = env.step(action)
                #print(info)
                fill_cnt=fill_cnt+1
                next_state=atari_processor.state_for_nn(observation)
                history_store.add_history(next_state)
                if(store_type=='vector'):
                    #print('vector')
                    next_nn_input=history_store.get_history_for_nn()
                else:
                    next_nn_input=history_store.get_history()
                if(first_step==True):
                    life=False
                    first_step=False
                    start_life=info
                else:
                    remain_life=info
                    if(remain_life!=start_life):
                        #print(info)
                        life=True
                        #print(life)
                        #print(done)
                        start_life=remain_life
                    else:
                        life=False
                        start_life=remain_life
                reward=atari_processor.process_reward(reward)
              
                self.append(nn_input, action, reward, next_nn_input,done,life)
                state=next_state
                if(done):
                    reset_flag=True
            #print(fill_cnt)
            if(fill_cnt>=(self.mem_size*ratio)):
                fill_flag=True


