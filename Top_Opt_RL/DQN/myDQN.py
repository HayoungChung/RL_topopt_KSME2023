# 관련 라이브러리 가져오기
import numpy as np 
from tensorflow.keras.optimizers import Adam # (if keras version 2.3.1)
import tensorflow as tf
from tensorflow.math import argmax
import tensorflow.keras as keras 
from tensorflow.keras import layers,models 
from collections import deque
import datetime

class QNet(keras.Model):
    def __init__(self, n_actions=None ,Increase=False):
        super(QNet, self).__init__()
        self.model = models.Sequential()

        self.model.add(layers.Conv2D(16,(3,3),padding='same',activation='relu'))
        self.model.add(layers.Conv2D(8,(3,3),padding='same',activation='relu'))
        self.model.add(layers.Conv2D(4,(3,3),padding='same',activation='relu'))
        self.model.add(layers.Conv2D(1,(3,3),padding='same',activation='relu'))
        self.model.add(layers.Flatten())
    
    def call(self, state):
        '''
        this returns Q for each state (forward pass)
        '''
        # x = self.model(state)
        # Q = x
        return self.model(state)
        
class ReplayBuffer():
    '''
    Experience Replay Buffer
    using deque (양 끝에서 삽입/삭제 모두 가능)
    input: memory size (#buffer), input dimensions (#elements)
    '''
    def __init__(self, mem_size, input_dims):
        self.state_size = mem_size * input_dims
        self.obs_buf = deque(maxlen=self.state_size) # observation
        self.obs2_buf = deque(maxlen=self.state_size) # next observation (float32)
        self.rew_buf = deque(maxlen=mem_size) # reward (float32)
        self.act_buf = deque(maxlen=mem_size) # action (int32)
        self.done_buf = deque(maxlen=mem_size) # done flag (bool)
        self.input_shape = input_dims # shape of state (N x N x 3)

    def store_transition(self, state, action, reward, state2, done):
        '''
        신규 SARS 버퍼 추가
        '''
        self.obs_buf.append(tuple(state.reshape(-1)))
        self.obs2_buf.append(tuple(state2.reshape(-1)))
        self.act_buf.append(action)
        self.rew_buf.append(reward)
        self.done_buf.append(done)

    def sample_buffer(self, batch_size):
        '''
        batch_size크기의 미니배치를 샘플링
        '''
        batch = np.random.choice(len(self.obs_buf), batch_size, replace=False)
        # states = self.obs_buf[batch]
        # actions = self.act_buf[batch]
        # rewards = self.rew_buf[batch]
        # states2 = self.obs2_buf[batch]
        # dones = self.done_buf[batch]
        # return states, actions, rewards, states2, dones

        states = np.array([self.state_memory[i] for i in batch])
        states2 = np.array([self.new_state_memory[i] for i in batch])
        actions = np.array([self.action_memory[i] for i in batch])
        rewards = np.array([self.reward_memory[i] for i in batch])
        dones = np.array([self.terminal_memory[i] for i in batch])

        return states.reshape(batch_size, *self.input_shape), actions, rewards, states2.reshape(batch_size, *self.input_shape), dones

    def __len__(self):
        return len(self.obs_buf)
    
# ===================================================================
'''
Agent 정의 (제일 중요!)
'''
class Agent():
    def __init__(self, env, opt, n_actions, input_dims, epsilon, epsilon_dec, epsilon_min, filename_save, filename_load, EX, EY):
        '''
        inputs: 
            env: env_primer of class TopOpt 
            opt: TopOpt options
            n_actions: N x N (N**2 요소 내 재료 제거)
            input_dims: [N, N, 3] (shape of states)
            epsilon: 초기 exploration rate
            epsilon_dec: exploration rate 감소율
            epsilon_min: 최소 exploration rate            
        '''
        self.env = env
        self.opt = opt
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.filename_save = filename_save
        self.filename_load = filename_load
        self.EX = EX
        self.EY = EY
        self.mem_size = 30000
        self.gamma  = 0.1

        self.memory = ReplayBuffer(opt.mem_size, self.input_dims)

        # Q network 초기화 =======
        self.q_eval = QNet(n_actions) 
        self.q_next = QNet(n_actions)
        self.q_eval.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        # just a formality
        self.q_next.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    def store_transition(self, state, action, reward, state2, done):
        '''
        버퍼에 SARS 추가
        '''
        self.memory.store_transition(state, action, reward, state2, done)
    
    def choose_action(self, state):
        '''
        Action 선정 (Epsilon-greedy)
        '''
        def greedy(Q_values):
            '''
            Greedy action selection
            '''
            return np.argmax(Q_values)

        def eps_greedy(Q_values, eps=0.1):
            '''
            Eps-greedy policy(엡실론 탐욕정책)
            Q_values: Q(s,a) for all a
            '''
            if np.random.uniform(0,1) < eps:
                return np.random.randint(len(Q_values))
            else:
                return np.argmax(Q_values)
        
        self.action_space = [i for i in range(self.n_actions)]
        if np.random.random() < self.epsilon: # exploration
            '''
            Void, BC, LC에 해당하는 action은 제외 (FEA 수행 불가능한 action)  
            '''
            Void=np.array(self.env.VoidCheck)
            BC=np.array(np.reshape(self.env.BC_state,(1,(self.EX*self.EY))))
            LC=np.array(np.reshape(self.env.LC_state,(1,(self.EX*self.EY))))
            Clear_List=np.where(Void==0)[0]
            BC_List=np.where(BC==1)[0]
            LC_List=np.where(LC==1)[0]
            self.action_space = [ele for ele in self.action_space if ele not in Clear_List]
            self.action_space = [ele for ele in self.action_space if ele not in BC_List]
            self.action_space = [ele for ele in self.action_space if ele not in LC_List]
            action = np.random.choice(self.action_space)
        else: # exploitation
            state=state.reshape(-1,self.EX,self.EY,3)
            actions = self.q_eval.call(state) # Q value for all actions
            action=argmax(actions, axis=1).numpy()[0]

    def learn(self):
        '''
        학습 (일반 DQN)
        '''

        # Set time for recording
        now = datetime.datetime.now()
        clock_time = str(now.year) + str(now.month) + str(now.day) + str(now.hour) + str(now.minute) + str(now.second)
        print("clock_time: ", clock_time)

        # Logging tensorflow
        log_dir = 'logs_ksme2023/'
        file_writer = tf.summary.create_file_writer(log_dir + clock_time)

        # Initialize hyperparameters
        NUM_EPSD    = 5000 # # of episodes
        RENDER_FLAG = True
        STEP_CNT    = 0 # step counter
        BATCH_SIZE  = 10
        MIN_BUFFER_SIZE = 1000
        UPDATE_FREQ = 5 # update frequency
        UPDATE_TARGET_FREQ = 100 # target network update frequency


        
        # Initialize Environment and Buffer
        self.env.reset_conditions()
        obs    = self.env.reset()  # obs: (N, N, 3): initial state
        # buffer = ReplayBuffer(self.opt.mem_size, self.input_dims)

        # Initialize Q network and target network
        # PASSED IN __init__()

        # initialize exploration rate
        eps = self.epsilon
        
        for ep in range(NUM_EPSD):
            g_new = 0
            done = False

            # Initialize episode
            # *** "Initialize block topology and Randomize loading cases" ***
            self.env.reset_conditions()

            while not done:


                # choose action (epsilon greedy)
                act = self.choose_action(obs, eps)

                # execute action and return new state, reward, done
                # *** "FEA to compute observation s_t" ***
                obs2, rew, done, _ = self.env.step(act)

                # print out 
                if RENDER_FLAG:
                    self.env.render()
                
                # store transition in buffer
                self.memory.store_transition(obs, act, rew, obs2, done)

                # update state
                obs = obs2
                g_rew += rew
                STEP_CNT += 1

                # if buffer is full and step count is multiple of UPDATE_FREQ, 
                # update "train" model
                
                if (len(self.buffer) > MIN_BUFFER_SIZE) and (STEP_CNT % UPDATE_FREQ == 0):
                    # sample batch from buffer
                    s_batch, a_batch, r_batch, s2_batch, d_batch = self.buffer.sample_buffer(BATCH_SIZE)

                    q_pred = self.q_eval(s_batch)
                    self.q_pred=q_pred
                    q_next = self.q_next(s2_batch)
                    q_target = q_pred.numpy()
                    max_actions = argmax(self.q_eval(s2_batch), axis=1)

                    # improve q_target
                    for idx, terminal in enumerate(d_batch):
                        q_target[idx, a_batch[idx]] = r_batch[idx] + \
                                self.gamma*q_next[idx, max_actions[idx]]*(1-int(d_batch[idx]))

                    # compute MSE
                    Loss=np.subtract(q_target,q_pred.numpy())
                    Loss=np.square(Loss)
                    Loss=Loss.mean()
                    # 이미 존재하는 모델에 대해 학습을 진행
                    self.q_eval.train_on_batch(s_batch, q_target) 

                    # epsilon greedy policy
                    if eps > self.epsilon_min:
                        eps -= self.eps_dec

                    # trainable variables
                    trainable_variables = self.q_eval.trainable_variables

                    # train model
                    self.q_eval.train_on_batch(s_batch, a_batch, r_batch, s2_batch, d_batch)

                    # update target network
                    self.q_next.set_weights(self.q_eval.get_weights())


                # for every 100
                # *** "update Q' by replacing auxiliary for main DNN weights" ***
                if (len(buffer) > MIN_BUFFER_SIZE) and (STEP_CNT % UPDATE_TARGET_FREQ == 0):
                    self.q_next.set_weights(self.q_eval.get_weights())           

    


    

        