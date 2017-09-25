import numpy as np
import copy
from numpy import array
from PIL import Image

class AtariProcessor:
    #Converts the observation images from environment to greyscale and downscales.
    def __init__(self, new_size):
        self.new_size=new_size
        #the length of vectorized frame
        self.length=new_size[0]*new_size[1]
    def state_for_mem(self, state):
        """Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        """
        #the input is the direct obervation from the environment
        #the return value is a processed single frame
        im=Image.fromarray(state)
        #print(im.mode)
        #print(im.format)
        im=im.resize((84,110),resample=1)
        im=im.convert('L')
        im=im.crop((0,26,84,110))
        
        output=array(im)
        #output=output.reshape((self.length))
        return output

    def state_for_nn(self, state):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """
        #the return value is a processed single frame
        #the input is the direction observation from the environment
        im=Image.fromarray(state)
        #print(im.mode)
        #print(im.format)
        im=im.resize((84,110),resample=1)
        im=im.convert('L')
        im=im.crop((0,26,84,110))

        output=array(im,dtype='float32')
        #output=output.reshape((self.length))
        return output


    def process_batch(self, samples):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        pass

    def process_reward(self, reward):
        """Clip reward between -1 and 1."""
        if reward>1:return 1
        if reward<-1:return -1
        return reward

class HistoryStore:
    """Keeps the last k states. This keeps the last k states as input for nn

    Useful for domains where you need velocities, but the state
    contains only positions.

    When the environment starts, this will just fill the initial
    sequence values with zeros k times.

    Parameters
    ----------
    history_length: int
      Number of previous states to prepend to state being processed.

    """

    def __init__(self, history_length=4,image_size=(84,84)):
        self.history_length=history_length
        self.image_size=image_size
        self.store=np.zeros((self.image_size[0],self.image_size[1],self.history_length),dtype='float32')
    def add_history(self, state):
        tmp=copy.deepcopy(self.store[:,:,1:self.history_length])
        new_store=np.zeros((self.image_size[0],self.image_size[1],self.history_length),dtype='float32')

        new_store[:,:,0:self.history_length-1]=tmp
        new_store[:,:,self.history_length-1]=state
        self.store=new_store
        #add the state to history store
        return True
    def get_history(self):
        #get currently stored history 84x84x4 in time order
        return self.store
    def get_history_for_nn(self):
        #get currently stored history as vector for nn, 1x28224, reshape in order F
        vectorized_history = np.reshape(self.store, (1, self.image_size[0]*self.image_size[1]*self.history_length),order='F')
        #vectorized_history=self.store.reshape((1, self.image_size[0]*self.image_size[1]*self.history_length))
        return vectorized_history
        #return vectorized_history.transpose()


    def reset(self):
        """Reset the history sequence.

        Useful when you start a new episode.
        """
        self.store=np.zeros((self.image_size[0],self.image_size[1],self.history_length),dtype='float32')
        return True

    def get_config(self):
        return {'history_length': self.history_length}


class HistoryStorev1:
    
    def __init__(self, history_length=4,image_size=(84,84)):
        self.pointer=0
        self.access_pointer=0
        self.history_length=history_length
        self.image_size=image_size
        self.store=np.zeros((self.image_size[0],self.image_size[1],self.history_length),dtype='float32')
    def add_history(self, state):
        if(self.pointer>=self.history_length): 
            self.pointer=self.pointer-self.history_length
        self.store[:,:,self.pointer]=state
        #print('add to pointer:',self.pointer)
        self.pointer=self.pointer+1
        #add the state to history store
        return True
    def get_history(self):
        #get currently stored history 84x84x4 returned in time order
        output=np.zeros((self.image_size[0],self.image_size[1],self.history_length),dtype='float32')
        self.access_pointer=self.pointer-1

        for i in range(self.history_length):
            output[:,:,self.history_length-i-1]=self.store[:,:,self.access_pointer]
            #print('acess_pointor:',self.access_pointer)
            #print('put to',self.history_length-i-1)
            self.access_pointer=self.access_pointer-1
        return output
        
        return self.store
    def get_history_for_nn(self):
        #get currently stored history as vector for nn 1x28224, F order
        output=np.zeros((self.image_size[0],self.image_size[1],self.history_length),dtype='float32')
        self.access_pointer=self.pointer-1

        for i in range(self.history_length):
            output[:,:,self.history_length-i-1]=self.store[:,:,self.access_pointer]
            #print('acess_pointor:',self.access_pointer)
            #print('put to',self.history_length-i-1)
            self.access_pointer=self.access_pointer-1
        vectorized_history=np.reshape(output,(1, self.image_size[0]*self.image_size[1]*self.history_length),order='F')
        return vectorized_history
        


    def reset(self):
        """Reset the history sequence.

        Useful when you start a new episode.
        """
        self.pointer=0
        self.store=np.zeros((self.image_size[0],self.image_size[1],self.history_length),dtype='float32')
        return True

    def get_config(self):
        return {'history_length': self.history_length}

