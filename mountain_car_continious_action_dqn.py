# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 17:53:58 2019

@author: David
"""

import gym
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from gym import wrappers


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
    def __init__(self,lr=0.001,gamma=0.9, num_action_bins = 100):
        self.env = gym.make('MountainCarContinuous-v0')
        self.sess = tf.Session()
        self.feat_transform = FeatureTransformer(env = self.env, n_comp = 500)        
        self.lr = lr
        self.num_action_bins = num_action_bins
        self.q_model = Q_model(lr=self.lr,num_state_inputs=self.feat_transform.num_features, num_action_outputs=self.num_action_bins,sess=self.sess)
        self.target_model = Q_model(lr=self.lr,num_state_inputs=self.feat_transform.num_features, num_action_outputs=self.num_action_bins,sess=self.sess)
        self.gamma = gamma
        self.action_splits = np.linspace(-1,1,self.num_action_bins + 1)
        
    
    def get_action_from_index(self,lower_idx):
        low = self.action_splits[lower_idx]
        high = self.action_splits[lower_idx + 1]
        return (low + high) / 2
    
    def get_index_from_action(self,action):
        return np.where(action >= self.action_splits)[0][-1]
    
    def get_transformed_state(self, state):     
        reshaped_state = np.reshape(state, (1,-1))
        transformed_state = self.feat_transform.transform(reshaped_state)        
        return transformed_state
    
    def get_truncated_x(self, x):
        if x >= 1:
            return 1
        elif x <= -1:
            return -1
        else:
            return x
        
    def pick_action_eps_greedy(self, eps, state):
        u = np.random.uniform()
        if u < eps:
            selected_action = np.random.uniform(low=-0.99999, high=0.99999)
            lower_idx = self.get_index_from_action(selected_action)
        else:
            input_features = self.get_transformed_state(state)
            predictions = self.q_model.predict(state_features = input_features)[0]
            lower_idx = np.argmax(predictions)
            selected_action = self.get_action_from_index(lower_idx)
            
        #selected_action = np.reshape(selected_action, (1,1))
        return selected_action, lower_idx
    
    def copy_q_model_target_model(self):
        for i in range(len(self.q_model.trainable_vars)):
            self.sess.run(self.target_model.trainable_vars[i].assign(self.q_model.trainable_vars[i]))
    
    def get_G_from_sars(self, s_a_r_s_prime):
        idx = random.randint(0, len(s_a_r_s_prime) - 1)
        s,a,r,s_prime = s_a_r_s_prime[idx]
        
        input_feat_s_prime = self.get_transformed_state(s_prime)
        Q_s_prime = self.target_model.predict(input_feat_s_prime)[0]
        G = r + self.gamma * np.max(Q_s_prime)
        return G
        
    
    def get_full_target(self, s,a,G):
        input_feat_s = self.get_transformed_state(s)
        Q_s = self.q_model.predict(input_feat_s)[0]      
        idx = self.get_index_from_action(a)
        
        full_target = Q_s.copy()
        full_target[idx] = G
        full_target_reshape = np.reshape(full_target, (1,-1))
        return full_target_reshape
    
    
    def play_episode_q_learning(self, eps=1, n=5, target_update_steps = 20):
        s = self.env.reset()
        done = False
        acc_r = 0
        s_a_r_s_prime = []
        step = 0
        while not done:
            #pick action
            a, lower_idx = self.pick_action_eps_greedy(eps = eps, state = s)
            
            #move  
            a_reshaped = np.reshape(a, (1,1))             
            s_prime, r, done, _ = self.env.step(a_reshaped)
            
            #update list
            if len(s_a_r_s_prime) < 5:
                s_a_r_s_prime.append((s,a,r,s_prime))
            else:
                del s_a_r_s_prime[0]
                s_a_r_s_prime.append((s,a,r,s_prime))               
                
            
            #get G
            if done:                
                G = r
            else:
                G = self.get_G_from_sars(s_a_r_s_prime)
                
            #get full target
            full_target = self.get_full_target(s,a,G)
            #print('full_target.shape = ',full_target.shape)
            #update Q model
            input_feat_s = self.get_transformed_state(s)
            self.q_model.update(input_feat_s,full_target)
            
            #update target model
            if (step % target_update_steps) == 0:
                self.copy_q_model_target_model()
            
            
            acc_r += r
            step += 1
            s = s_prime
            
        return acc_r
    
    def train_using_q_learning(self, N, M, eps, n, target_update_steps):
        rewards = []
        for i in range(N):
            rew = []
            for j in range(M):
                r = self.play_episode_q_learning(eps=eps, n=n, target_update_steps = target_update_steps)
                rew.append(r)                
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
            a,_ = self.pick_action_eps_greedy(eps=eps,state=s)
            s_prime, r, done, _ = self.env_record.step(a)
            s = s_prime
            acc_r += r
        return acc_r
    
    def test(self,N):       
        self.env_record = wrappers.Monitor(self.env, 'mountain_car_continuous_dqn')
        rewards = []
        for i in range(N):
            r = self.play_episode()
            rewards.append(r)
            print('test i = ',i)
        mean_r = np.mean(np.array(rewards))
        print('mean testing reward = ', mean_r)
   
    
    

class Q_model():
    def __init__(self,lr,num_state_inputs, num_action_outputs, sess):
        self.lr = lr
        self.num_state_inputs = num_state_inputs
        self.num_action_outputs = num_action_outputs
        self.sess = sess
        self.build_model()
        self.initialize_variables()
        
        
    def build_model(self):
        self.Input = tf.placeholder(dtype='float32', shape=(None, self.num_state_inputs), name='Policy_input')        
        self.target = tf.placeholder(dtype='float32', shape=(None, self.num_action_outputs), name = 'target')
        
        self.W1 = tf.Variable(tf.constant(0, shape=[self.num_state_inputs, self.num_action_outputs], dtype='float32'), name = 'W1')
        self.b1 = tf.Variable(tf.constant(0, shape=[self.num_action_outputs], dtype='float32'), name = 'b1')
        
        self.trainable_vars = [self.W1, self.b1]
                
        self.output = tf.add(tf.matmul(self.Input, self.W1), self.b1)
        self.loss = tf.reduce_mean(tf.squared_difference(self.output, self.target))
        
        self.optimize_step = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss)
        self.init_op = tf.global_variables_initializer()
    
    def initialize_variables(self):       
        self.sess.run(self.init_op)
        
    
    def predict(self, state_features):
        return self.sess.run(self.output, feed_dict = {self.Input: state_features})
        
    
    def update(self, state_features, target):
        self.sess.run(self.optimize_step, feed_dict = {self.Input: state_features, self.target: target})
    
    

if __name__ == '__main__':    
    agent = Agent()
    agent.train_using_q_learning(N=10, M=10, eps=1, n=5, target_update_steps=20)
    agent.test(N=20)
    agent.sess.close()
