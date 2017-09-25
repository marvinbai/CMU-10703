import numpy as np
import scipy as sp
import gym
import keras
import os
import time
from deeprl_hw3.imitation import load_model, wrap_cartpole, test_cloned_policy,generate_expert_training_data,process_state,LossHistory,wrap_cartpole
from keras.optimizers import Adam
from keras import backend as K
t=time.time()
    
folder_path = './Q2_DAGGER_at_'+str(t)+'/'
print(folder_path)
directory = os.path.dirname(folder_path)
if not os.path.exists(directory):
    os.makedirs(directory)
  
EXPERT_EPISODES=1
TRAIN_EPOCHS=100
ADAM_LR=0.001
    
EXPERT_EPISODES_LIST=[1,10,50,100]
TRAIN_EPOCHS_LIST=[50,100,150,200]
    
env=gym.make('CartPole-v0')
env2=gym.make('CartPole-v0')
env_hard=wrap_cartpole(env2)
def main():
    for j in range(len(TRAIN_EPOCHS_LIST)):
        EXPERT_EPISODES=0
        TRAIN_EPOCHS=TRAIN_EPOCHS_LIST[j]
        run_Q2(env,env_hard,EXPERT_EPISODES,TRAIN_EPOCHS,folder_path)
    
    
def run_Q2(env,env_hard,EXPERT_EPISODES,TRAIN_EPOCHS,folder_path):
       
    file_path=folder_path+'Q2_'+str(EXPERT_EPISODES)+'_'+str(TRAIN_EPOCHS)+'.txt'
    f = open(file_path, 'w')
    f.write('Parameters:\n')
    f.write('EXPET_EPISODES:'+str(EXPERT_EPISODES)+'\n')
    f.write('TRAIN_EPOCHS:'+str(TRAIN_EPOCHS)+'\n')
    
    
    
    #test all parameters
    
    expert=load_model('CartPole-v0_config.yaml','CartPole-v0_weights.h5f')
    learner=load_model('CartPole-v0_config.yaml',None)
    adam=Adam()
    expert.compile(adam,'binary_crossentropy',metrics=['accuracy'])
    learner.compile(adam,'binary_crossentropy',metrics=['accuracy'])
    print('Generate initial data from learner')
    data,_=generate_expert_training_data(learner,env,num_episodes=1,render=False)
    print('Qurey expert for labels ')
    q_values=expert.predict(data)
    labels=np.argmax(q_values, axis=1)
    onehot_labels=np.zeros((labels.shape[0],2))
    for i in range(labels.shape[0]):
        onehot_labels[i,labels[i]]=1
    #print(onehot_labels)
    #print(onehot_labels.shape)
    print('Expert qurey is ready. Start to train learner with epoch num:',TRAIN_EPOCHS)

    history = LossHistory()
    train_cnt=0
    while train_cnt<TRAIN_EPOCHS:
        learner.fit(data,onehot_labels,epochs=1,callbacks=[history])
        #generate new data for DAAGER
        # use the same function as generate expert, but using the learner model
        new_data,_=generate_expert_training_data(learner,env,num_episodes=1,render=False)
        print('Qurey expert for labels ')
        new_q_values=expert.predict(new_data)
        new_labels=np.argmax(new_q_values, axis=1)
        new_onehot_labels=np.zeros((new_labels.shape[0],2))
        for i in range(new_labels.shape[0]):
            new_onehot_labels[i,new_labels[i]]=1
        data=np.vstack((data,new_data))
        onehot_labels=np.vstack((onehot_labels,new_onehot_labels))
        print(onehot_labels.shape)
        train_cnt=train_cnt+1
    
    weights_path=folder_path+'Q2_'+str(EXPERT_EPISODES)+'_'+str(TRAIN_EPOCHS)+'.h5'
    learner.save_weights(weights_path)
    
    print('Test expert in normal env.........................................')
    expert_reward_summary,expert_reward_avg,expert_reward_std=test_cloned_policy(env,expert, num_episodes=100, render=False)
    print('Test learner in normal env.........................................')
    learner_reward_summary,learner_reward_avg,learner_reward_std=test_cloned_policy(env,learner, num_episodes=100, render=False)      
    
    print('Test expert in hard Env.........................................')
    hard_expert_reward_summary,hard_expert_reward_avg,hard_expert_reward_std=test_cloned_policy(env_hard,expert, num_episodes=100, render=False)
    print('Test learner in hard Env.........................................')
    hard_learner_reward_summary,hard_learner_reward_avg,hard_learner_reward_std=test_cloned_policy(env_hard,learner, num_episodes=100, render=False)
    
    f.write('Expert Test in Normal Env:\n')
    f.write(str(expert_reward_avg)+'    '+str(expert_reward_std)+'\n')
    f.write('Learner Test in Normal Env:\n')
    f.write(str(learner_reward_avg)+'    '+str(learner_reward_std)+'\n')
    f.write('Expert Test in Hard Env:\n')
    f.write(str(hard_expert_reward_avg)+'    '+str(hard_expert_reward_std)+'\n')
    f.write('Learner Test in Hard Env:\n')
    f.write(str(hard_learner_reward_avg)+'    '+str(hard_learner_reward_std)+'\n')
    f.write('Learner Training  History:\n')
    for i in range(TRAIN_EPOCHS):
      f.write(str(history.losses[i])+'    '+str(history.accues[i])+'\n')
    
    f.write('Evaluate History:\n')
    for i in range(100):
      f.write(str(expert_reward_summary[i])+'    '+str(learner_reward_summary[i])+'    '+str(hard_expert_reward_summary[i])+'    '+str(hard_learner_reward_summary[i])+'\n')
    
    f.close()

if __name__ == '__main__':
    main()
