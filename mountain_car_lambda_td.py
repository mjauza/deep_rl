
import gym
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from gym import wrappers
#env = gym.make('MountainCar-v0')

class FeatureTransformer():
    def __init__(self,env, n_comp = 500):
        N = 10000
        observation_examples = np.array([env.observation_space.sample() for x in range(N)])
        scaler = StandardScaler()
        scaler.fit(observation_examples)
        
        featurizer = FeatureUnion([('rbf1', RBFSampler(gamma=5.0, n_components=n_comp)),
                                  ('rbf2', RBFSampler(gamma=2.0, n_components=n_comp)),
                                  ('rbf3', RBFSampler(gamma=1.0, n_components=n_comp)),
                                  ('rbf4', RBFSampler(gamma=0.5, n_components=n_comp))])
        featurizer.fit(scaler.transform(observation_examples))
        
        self.scaler = scaler
        self.featurizer = featurizer
        self.num_features = 4*n_comp
        
    def transform(self,x):
        return self.featurizer.transform(self.scaler.transform(x))

class Agent():
    def __init__(self,lr=0.01,gamma=0.9):
        self.env = gym.make('MountainCar-v0')
        self.feat_transform = FeatureTransformer(env = self.env, n_comp = 500)
        self.num_actions = 3
        self.Q_model = Model(lr = lr, num_actions = self.num_actions, num_input_features=self.feat_transform.num_features)
        self.gamma = gamma
        
    def pick_action_eps_greedy(self,eps,state):
        u = np.random.uniform()
        if u < eps:
            selected_action = random.randint(0,self.num_actions-1)
        else:
            state = np.reshape(state, (1,-1))
            input_features = self.feat_transform.transform(state)
            predicted_Q = self.Q_model.predict(input_features)[0]
            #print('predicted Q shape = ',predicted_Q.shape)
            selected_action = np.argmax(predicted_Q)
        return selected_action            
            
    def get_full_target(self,state,action,target):
        state_reshaped = np.reshape(state,(1,-1))
        input_features_state_reshaped = self.feat_transform.transform(state_reshaped)
        full_target = self.Q_model.predict(input_features_state_reshaped)
        full_target[:,int(action)] = target
        return full_target
        
    def update_Q_n_step(self, s,a,s_prime,reward_list):
        n = len(reward_list)
        #get full target
        s_prime_reshaped = np.reshape(s_prime, (1,-1))
        input_features_s_prime = self.feat_transform.transform(s_prime_reshaped)
        Q_s_prime = self.Q_model.predict(input_features_s_prime)[0]
        discounted_rewards =  np.array(reward_list)*(self.gamma**np.arange(0,n))
        target = discounted_rewards + (self.gamma**n) * np.max(Q_s_prime)
        full_target = self.get_full_target(state=s,action=a,target=target)
        
        #update model
        s_reshaped = np.reshape(s, (1,-1))
        input_features_s = self.feat_transform.transform(s_reshaped)
        self.Q_model.update(input_features=input_features_s, target = full_target)
    
    def get_e_t(self,lam,e_t_1,s):
        e_t_1_W = e_t_1[0]
        e_t_1_b = e_t_1[1]
        s_reshaped = np.reshape(s, (1,-1))
        input_features = self.feat_transform.transform(s_reshaped)
        dW1 = self.Q_model.get_gradient_W(input_features)
        db1 = self.Q_model.get_gradient_b(input_features)        
        et_W = dW1 + self.gamma * lam * e_t_1_W
        et_b = db1 + self.gamma * lam * e_t_1_b
        return et_W, et_b
    
    def get_delta_t(self,r,s,a,s_prime):
        s_prime_reshaped = np.reshape(s_prime, (1,-1))
        s_reshaped = np.reshape(s, (1,-1))
        input_features_s_prime = self.feat_transform.transform(s_prime_reshaped)
        input_features_s = self.feat_transform.transform(s_reshaped)
        Q_s_prime = self.Q_model.predict(input_features_s_prime)
        Q_s_a = self.Q_model.predict(input_features_s)[0][int(a)]
        delta_t = r + self.gamma * np.max(Q_s_prime) - Q_s_a
        return delta_t
    
    def play_episode_q_learning_td_lambda(self,eps,lam):
        #initialize env
        s = self.env.reset()
        done = False
        et_W = 0
        et_b = 0
        et = [et_W,et_b]
        acc_r = 0
        
        while not done:
            #pick action eps greedy
            a = self.pick_action_eps_greedy(eps=eps,state=s)
            
            #move
            s_prime, r, done, _ = self.env.step(a)
            
            #get eligability trace
            et = self.get_e_t(lam = lam, e_t_1 = et, s = s)
            
            #get delta t
            delta_t = self.get_delta_t(r,s,a,s_prime)
            
            #define update value 
            delta_t_e_t_W = delta_t * et[0]
            delta_t_e_t_b = delta_t * et[1]
            
            #update Q
            self.Q_model.update_W(delta_t_e_t_W)
            self.Q_model.update_b(delta_t_e_t_b)
            
            s = s_prime
            acc_r += r
            
        return acc_r
    
    def train_q_learning_td_lambda(self,N,M,eps,lam=0.5):
        rewards = []
        for i in range(N):
            rew = []
            for j in range(M):
                r = self.play_episode_q_learning_td_lambda(eps=eps,lam=lam)
                rew.append(r)
                print('j = ',j)
            rewards.append(np.mean(np.array(rew)))
            print('i = ',i)
        
        fig = plt.figure()
        plt.plot(rewards)
        plt.title('rewards')
        fig.show()
        
    
    def play_episode_q_learning_n_step(self,eps,n):
        #initialize env
        s = self.env.reset()
        done = False
        acc_r = 0
        s_a_r_dic = {}
        #s_a_s_prime = {}
        while not done:
            #pick action eps gredy
            a = self.pick_action_eps_greedy(eps=eps,state=s)
            
            #move
            s_prime, r, done, _ = self.env.step(a)
            
            s_hash = tuple(s.tolist())
            a_hash = a
            if (s_hash,a_hash) not in s_a_r_dic:
                s_a_r_dic[(s_hash,a_hash)] = [r]
            else:
                s_a_r_dic[(s_hash,a_hash)].append(r)
            
            list_to_pop = []
            for (s_david,a_david) in s_a_r_dic:
                if done:
                    s_david_np = np.array(s_david)
                    a_david_np = np.array(a_david)
                    self.update_Q_n_step(s=s_david_np,a=a_david_np,s_prime=s_prime,reward_list=s_a_r_dic[(s_david,a_david)])                    
                                        
                elif len(s_a_r_dic[(s_david,a_david)]) == n:
                    s_david_np = np.array(s_david)
                    a_david_np = np.array(a_david)
                    self.update_Q_n_step(s=s_david_np,a=a_david_np,s_prime=s_prime,reward_list=s_a_r_dic[(s_david,a_david)])
                    list_to_pop.append((s_david,a_david))
            
            for (s_david,a_david) in list_to_pop:
                s_a_r_dic.pop((s_david,a_david),None)

            
            s = s_prime
            acc_r += r
            
        return acc_r
    
    def train_q_learning_n_step(self,N,M,eps,n=3):
        rewards = []
        for i in range(N):
            rew = []
            for j in range(M):
                r = self.play_episode_q_learning_n_step(eps=eps,n=n)
                rew.append(r)
                print('j = ',j)
            rewards.append(np.mean(np.array(rew)))
            print('i = ',i)
        
        fig = plt.figure()
        plt.plot(rewards)
        plt.title('rewards')
        fig.show()
        
    def play_episode(self,eps=0):
        s = self.env_record.reset()
        done = False
        acc_r = 0
        while not done:
            a = self.pick_action_eps_greedy(eps=eps,state=s)
            s_prime, r, done, _ = self.env_record.step(a)
            s = s_prime
            acc_r += r
        return acc_r
            
    
    def test(self,N):
        self.env_record = wrappers.Monitor(self.env, 'mountain_car_td_lambda')
        rewards = []        
        for i in range(N):
            r = self.play_episode()
            rewards.append(r)
        mean_r = np.mean(np.array(rewards))
        print('mean testing reward = ', mean_r)

class Model():
    def __init__(self,lr,num_actions,num_input_features):
        self.lr = lr
        self.num_actions = num_actions
        self.num_input_features = num_input_features
        self.initialize_model_var_and_sess()
    
    def build_model(self):
        Input = tf.placeholder(tf.float32, shape = [None, self.num_input_features], name='input')
        Label = tf.placeholder(tf.float32, shape = [None,self.num_actions], name='label')
                
        #W1 = tf.Variable(tf.random_normal([self.num_input_features, self.num_actions], stddev=1), name='W1')
        #b1 = tf.Variable(tf.random_normal([self.num_actions]), name='b1')
        
        W1 = tf.Variable(tf.constant(0, shape=[self.num_input_features, self.num_actions], dtype='float32'), name='W1')
        b1 = tf.Variable(tf.constant(0, shape = [self.num_actions], dtype='float32'), name='b1')
        
        output = tf.add(tf.matmul(Input, W1),b1)
                
        loss = tf.reduce_mean(tf.squared_difference(output, Label))
        optimize_step = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(loss)
        init_op = tf.global_variables_initializer()
        
        dW1,db1 = tf.gradients(output, [W1,b1]) 
        
        self.W1 = W1
        self.b1 = b1
        self.dW1 = dW1
        self.db1 = db1
        self.Label = Label
        self.Input = Input
        self.output = output
        self.loss = loss
        self.optimize_step = optimize_step
        self.init_op = init_op
    
    def initialize_model_var_and_sess(self):
        self.build_model()        
        self.sess = tf.Session()
        self.sess.run(self.init_op)
    
    def update(self, input_features, target):
        target = np.reshape(target,(1,-1))
        input_features = np.reshape(input_features, (1,-1))
        self.sess.run(self.optimize_step, feed_dict = {self.Input: input_features, self.Label:target})
    
    def predict(self, input_features):
        input_features = np.reshape(input_features, (1,-1))
        prediction = self.sess.run(self.output, feed_dict = {self.Input: input_features})
        return prediction
    
    def get_gradient_W(self,input_features):
        input_features = np.reshape(input_features, (1,-1))
        return self.sess.run(self.dW1, feed_dict = {self.Input: input_features})
    
    def get_gradient_b(self,input_features):
        input_features = np.reshape(input_features, (1,-1))
        return self.sess.run(self.db1, feed_dict = {self.Input: input_features})
    
    def update_W(self, delta_t_e_t):
        value = tf.convert_to_tensor(self.lr*delta_t_e_t)
        tf.assign_add(self.W1, value)
    
    def update_b(self, delta_t_e_t):
        value = tf.convert_to_tensor(self.lr*delta_t_e_t)
        tf.assign_add(self.b1, value)

    
    def close_sess(self):
        self.sess.close()
    
    
if __name__ == '__main__':
    agent = Agent()
    #agent.train_q_learning_n_step(N=200,M=10,eps=1)
    agent.train_q_learning_td_lambda(N=10,M=1,eps=1)
    agent.test(N=100)
    agent.Q_model.close_sess()
        
        