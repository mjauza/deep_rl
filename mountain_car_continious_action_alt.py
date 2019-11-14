# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 12:28:04 2019

@author: David
"""



import gym
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers
import os
from os import listdir
from os.path import isfile, join


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
    def __init__(self,lr=0.001,gamma=0.9,restore=True):
        self.env = gym.make('MountainCarContinuous-v0')
        self.feat_transform = FeatureTransformer(env = self.env, n_comp = 500)        
        self.lr = lr
        self.step = 0
        self.policy_model = Policy_model(lr=self.lr, num_inputs = self.feat_transform.num_features, step = self.step)
        self.value_model = Value_model(lr = self.lr, num_inputs = self.feat_transform.num_features, step = self.step)
        self.gamma = gamma
        
        if restore:            
            self.policy_model.restore__latest_session()
            self.value_model.restore__latest_session()
        
    def get_reshaped_transformed_state(self, state):
        reshaped_state = np.reshape(state, (1,-1))
        return self.feat_transform.transform(reshaped_state)
    
    def get_truncated_x(self, x):
        if x >= 1:
            return 1
        elif x <= -1:
            return -1
        else:
            return x
    
    def transform_target(self,target):
        return np.tanh(target)
        #return 10000*target
    
    def get_sample(self,p, alpha, beta):
        x = np.random.binomial(1,p,1)
        beta_sample = np.random.beta(a=alpha,b=beta,size = 1)
        return x*beta_sample - (1-x)*beta_sample
    
    def pick_action_eps_greedy(self, eps, state):
        u = np.random.uniform()
        if u < eps:
            v1 = np.random.uniform(low=0.001,high=0.999, size = 1)
            v2 = np.random.uniform(low=-0.999,high=-0.001, size = 1)
            p = np.random.uniform()
            if p < 1/2:
                selected_action = v1
            else:
                selected_action = v2
            
        else:
            input_features = self.get_reshaped_transformed_state(state)
            
            p,alpha,beta = self.policy_model.predict_p_alpha_beta(input_features)
            alpha = alpha[0]
            beta = beta[0]
            p = p[0]
           
            selected_action = self.get_sample(p, alpha, beta)
        selected_action = np.array(selected_action)
        selected_action = np.reshape(selected_action,(1,1))
        return selected_action
    
    def play_episode_actor_critic(self, eps=1):
        s = self.env.reset()
        done = False
        acc_r = 0
        david = 0
        print_log1 = True
        print_log2 = True
        print_both = False
        #f = open("log.txt", "a")        
        
        while not done:
            input_feat = self.get_reshaped_transformed_state(s)
            p,alpha,beta = self.policy_model.predict_p_alpha_beta(input_feat)            
                
                
            #pick action
            a = self.pick_action_eps_greedy(eps = eps, state = s)
            
            #move            
            s_prime, r, done, _ = self.env.step(a)
            
            #get G
            if done:                
                G = np.array([r])
            else:
                input_feat_s_prime = self.get_reshaped_transformed_state(s_prime)
                V_s_prime = self.value_model.predict(input_feat_s_prime)[0]
                G = np.array(r + self.gamma * V_s_prime)
            
            #get advantage and action
            input_feat_s = self.get_reshaped_transformed_state(s)
            V_s = self.value_model.predict(input_feat_s)[0]
            advantage = [(G - V_s).tolist()]
            
            transformed_adv = self.transform_target(G - V_s).reshape((1,1))
            action = a
            target = [G.tolist()]            
            transformed_target = self.transform_target(G).reshape((1,1))
            
            
            self.value_model.update(input_features=input_feat_s, target=transformed_target)
            self.policy_model.update(input_features=input_feat_s, advantage=transformed_adv, selected_action=action)
            
            acc_r += r
            s = s_prime
            david += 1
        
        #f.close()
        return acc_r
    
    def train_using_actor_critic(self, N,M,eps):
        rewards = []
        for i in range(N):
            rew = []
            for j in range(M):
                r = self.play_episode_actor_critic(eps=eps)
                rew.append(r)
                self.policy_model.step += 1
                self.value_model.step += 1
                #print('j = ',j)
            rewards.append(np.mean(np.array(rew)))
            print('i = ',i)
        
        self.policy_model.save_model()
        self.value_model.save_model()
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
        self.env_record = wrappers.Monitor(self.env, 'mountain_car_continuous_actor_critic_alt')
        rewards = []
        for i in range(N):
            r = self.play_episode()
            rewards.append(r)
            print('test i = ',i)
        mean_r = np.mean(np.array(rewards))
        print('mean testing reward = ', mean_r)
   
    
    

class Value_model():
    def __init__(self, lr, num_inputs , step):
        self.lr = lr
        self.num_inputs = num_inputs
        self.build_model()
        self.initialze_variables_and_sess()
        self.initialize_saver()
        self.step = step
        
        
    def build_model(self):
        self.Input = tf.placeholder(dtype='float32', shape=[None, self.num_inputs] ,name='V_input')
        self.target = tf.placeholder(dtype='float32', shape=[None, 1], name='V_target')
        
        
        self.W1 = tf.Variable(tf.constant(0, shape=[self.num_inputs, 1], dtype='float32'), name = 'W1')
        self.b1 = tf.Variable(tf.constant(0, shape=[1], dtype='float32'), name = 'b1')
        
        
        net = tf.add(tf.matmul(self.Input, self.W1), self.b1)
        
        self.output = net
        
        self.loss = tf.reduce_mean(tf.squared_difference(self.output, self.target))        
        self.optimize_step = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss)
        self.init_op = tf.global_variables_initializer()        
    
    def initialze_variables_and_sess(self):
        self.sess = tf.Session()
        self.sess.run(self.init_op)
        
    def predict(self, input_features):
        return self.sess.run(self.output, feed_dict = {self.Input: input_features})
    
    def update(self, input_features, target):
        self.sess.run(self.optimize_step, feed_dict = {self.Input: input_features, self.target: target})
    
    def close_sess(self):
        self.sess.close()
    
    def maybe_make_ckpt_dir(self,directory='./checkpoint_value'):
        if not os.path.isdir(directory):
            os.mkdir(directory)
    
    def initialize_saver(self):
        self.saver = tf.train.Saver()
    
    def save_model(self,directory='./checkpoint_value'):
        self.maybe_make_ckpt_dir(directory=directory)
        filename = directory + '/' + 'value_model'
        self.saver.save(self.sess, filename, global_step= self.step)
        
    def get_latest_checkpoint(self,directory='./checkpoint_value'):
        mypath = directory
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        only_meta = [f for f in onlyfiles if f[-4:] == 'meta']
        maxi = -1
        #print(only_meta)
        for f in only_meta:
            print(f[:-5])
            print(f[:-5].split('-'))
            num = int(f[:-5].split('-')[1])
            if num > maxi:
                maxi = num
                filename = f
        
        self.step = maxi + 1
        self.latest_metafile = filename
    
    def restore__latest_session(self, directory='./checkpoint_value'):
        self.sess = tf.Session()
        self.get_latest_checkpoint(directory=directory)
        print('Restoring from model '+self.latest_metafile)
        filename = directory + '/' + self.latest_metafile
        self.saver = tf.train.import_meta_graph(filename)
        ckpt =  tf.train.latest_checkpoint(directory)
        self.saver.restore(self.sess,ckpt)
        self.sess.run(self.init_op)
        

class Policy_model():
    def __init__(self,lr, num_inputs, step):
        self.lr = lr
        self.num_inputs = num_inputs
        self.num_outputs = 2
        self.build_model()
        self.initialize_variables_and_sess()
        self.initialize_saver()
        self.step = step
        
        
    def build_model(self):
        self.Input = tf.placeholder(dtype='float32', shape=[1, self.num_inputs], name='Policy_input')
        self.advantage = tf.placeholder(dtype='float32', shape=[1, 1], name='Policy_advantage')
        self.selected_action = tf.placeholder(dtype='float32', shape=[1,1], name='Policy_action')
        
        
        self.W1_p = tf.Variable(tf.constant(0.5, shape=[self.num_inputs, 1], dtype='float32'), name = 'W1_p')
        self.b1_p = tf.Variable(tf.constant(0.5, shape=[1], dtype='float32'), name = 'b1_p')
        
        self.W1_alpha = tf.Variable(tf.constant(10, shape=[self.num_inputs, 1], dtype='float32'), name = 'W1_alpha')
        self.b1_alpha = tf.Variable(tf.constant(10, shape=[1], dtype='float32'), name = 'b1_alpha')
        
        self.W1_beta = tf.Variable(tf.constant(5, shape=[self.num_inputs, 1], dtype='float32'), name = 'W1_beta')
        self.b1_beta = tf.Variable(tf.constant(5, shape=[1], dtype='float32'), name = 'b1_beta')
                
        self.net_p = tf.add(tf.matmul(self.Input, self.W1_p), self.b1_p)        
        self.net_alpha = tf.add(tf.matmul(self.Input, self.W1_alpha), self.b1_alpha)
        self.net_beta = tf.add(tf.matmul(self.Input, self.W1_beta), self.b1_beta)
               
        
        p = tf.math.sigmoid(self.net_p)
        
        self.p = tf.clip_by_value(p, 0.0001, 0.9999)
        
        self.alpha = tf.nn.softplus(self.net_alpha) + 1e-4
        self.beta = tf.nn.softplus(self.net_beta) + 1e-4
               
        alpha = tf.reshape(self.alpha, shape=(1,))
        beta = tf.reshape(self.beta, shape=(1,))
        beta_dist = tf.distributions.Beta(alpha,beta)
        
        beta_pdf = beta_dist.prob(tf.math.abs(self.selected_action))
        self.beta_pdf = tf.clip_by_value(beta_pdf, 0.0001, 0.9999)
        
        self.david = tf.multiply(self.p,self.beta_pdf)
        self.david1 = tf.multiply(1 - self.p,self.beta_pdf)
        self.prob = tf.cond(tf.reshape(self.selected_action, []) > 0, lambda: tf.multiply(self.p,self.beta_pdf) , lambda: tf.multiply(1 - self.p,self.beta_pdf))
        
        self.log_probs = tf.math.log(self.prob)
                
        self.cost = -tf.reduce_mean(tf.multiply(self.advantage, self.log_probs))
        
        self.optimize_step = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.cost)
        self.init_op = tf.global_variables_initializer()
    
    def initialize_variables_and_sess(self):
        self.sess = tf.Session()
        self.sess.run(self.init_op)
        
    
    def predict_p_alpha_beta(self, input_features):
        p, alpha, beta = self.sess.run([self.p, self.alpha, self.beta], feed_dict={self.Input: input_features})
        return p, alpha, beta
        
    
    def update(self, input_features, advantage, selected_action):
        self.sess.run(self.optimize_step, feed_dict = {self.Input: input_features, self.advantage: advantage, self.selected_action: selected_action})
    
    def get_some_numbers(self, input_features, advantage, selected_action):
        prob, log_prob, cost, beta_pdf, david, david1,net_p,net_alpha,net_beta = self.sess.run([self.prob, self.log_probs, self.cost,self.beta_pdf,self.david, self.david1,
                                                                       self.net_p,self.net_alpha,self.net_beta ],  
                                             feed_dict = {self.Input: input_features, self.advantage: advantage, self.selected_action: selected_action})
        return prob, log_prob, cost, beta_pdf, david, david1,net_p,net_alpha,net_beta
    
    def close_sess(self):
        self.sess.close()
        
    def maybe_make_ckpt_dir(self,directory='./checkpoint_policy'):
        if not os.path.isdir(directory):
            os.mkdir(directory)
    
    def initialize_saver(self):
        self.saver = tf.train.Saver()
    
    def save_model(self,directory='./checkpoint_policy'):
        self.maybe_make_ckpt_dir(directory=directory)
        filename = directory + '/' + 'policy_model'
        self.saver.save(self.sess, filename, global_step= self.step)
        
    def get_latest_checkpoint(self,directory='./checkpoint_policy'):
        mypath = directory
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        only_meta = [f for f in onlyfiles if f[-4:] == 'meta']
        maxi = -1
        #print(only_meta)
        for f in only_meta:
            print(f[:-5])
            print(f[:-5].split('-'))
            num = int(f[:-5].split('-')[1])
            if num > maxi:
                maxi = num
                filename = f
        
        self.step = maxi + 1
        self.latest_metafile = filename
    
    def restore__latest_session(self, directory='./checkpoint_policy'):
        self.sess = tf.Session()
        self.get_latest_checkpoint(directory=directory)
        print('Restoring from model '+self.latest_metafile)
        filename = directory + '/' + self.latest_metafile
        self.saver = tf.train.import_meta_graph(filename)
        ckpt =  tf.train.latest_checkpoint(directory)
        self.saver.restore(self.sess,ckpt)
        self.sess.run(self.init_op)

if __name__ == '__main__':
    agent = Agent(restore = True,lr=0.0001)
    agent.train_using_actor_critic(N=50,M=10,eps=1)
    agent.test(N=80)
    agent.value_model.close_sess()
    agent.policy_model.close_sess()
