# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 13:55:38 2019

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
    def __init__(self,lr=0.001,gamma=0.9):
        self.env = gym.make('MountainCarContinuous-v0')
        self.feat_transform = FeatureTransformer(env = self.env, n_comp = 500)        
        self.lr = lr
        self.policy_model = Policy_model(lr=self.lr, num_inputs = self.feat_transform.num_features)
        self.value_model = Value_model(lr = self.lr, num_inputs = self.feat_transform.num_features)
        self.gamma = gamma
        
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
        
    def pick_action_eps_greedy(self, eps, state):
        u = np.random.uniform()
        if u < eps:
            selected_action = self.get_truncated_x(np.random.normal())
        else:
            input_features = self.get_reshaped_transformed_state(state)
            
            mu,sigma = self.policy_model.predict_mu_sigma(input_features)
            print('mu = ', mu)
            print('sigma = ',sigma)
            mu = mu[0]
            sigma = sigma[0] + 0.00001
            
            if sigma < 0:                
                sigma = 0.0001
            #print('sigma = ',sigma)    
            selected_action = self.get_truncated_x(np.random.normal(loc=mu, scale=sigma))
        selected_action = np.array(selected_action)
        selected_action = np.reshape(selected_action,(1,1))
        return selected_action
    
    def play_episode_actor_critic(self, eps=1):
        s = self.env.reset()
        done = False
        acc_r = 0
        while not done:
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
            action = a
            target = [G.tolist()]
            
            
            
            #update            
            self.value_model.update(input_features=input_feat_s, target=target)
            self.policy_model.update(input_features=input_feat_s, advantage=advantage, selected_action=action)
            
            acc_r += r
            s = s_prime
            
        return acc_r
    
    def train_using_actor_critic(self, N,M,eps):
        rewards = []
        for i in range(N):
            rew = []
            for j in range(M):
                r = self.play_episode_actor_critic(eps=eps)
                rew.append(r)
                #print('j = ',j)
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
        self.env_record = wrappers.Monitor(self.env, 'mountain_car_continuous_actor_critic')
        rewards = []
        for i in range(N):
            r = self.play_episode()
            rewards.append(r)
            print('test i = ',i)
        mean_r = np.mean(np.array(rewards))
        print('mean testing reward = ', mean_r)
   
    
    

class Value_model():
    def __init__(self, lr, num_inputs):
        self.lr = lr
        self.num_inputs = num_inputs
        self.build_model()
        self.initialze_variables_and_sess()
        
        
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
        

class Policy_model():
    def __init__(self,lr, num_inputs):
        self.lr = lr
        self.num_inputs = num_inputs
        self.num_outputs = 2
        self.build_model()
        self.initialize_variables_and_sess()
        
    def build_model(self):
        self.Input = tf.placeholder(dtype='float32', shape=[None, self.num_inputs], name='Policy_input')
        self.advantage = tf.placeholder(dtype='float32', shape=[None, 1], name='Policy_advantage')
        self.selected_action = tf.placeholder(dtype='float32', shape=[None,1], name='Policy_action')
        
        #n1 = 100
        self.W1 = tf.Variable(tf.constant(0, shape=[self.num_inputs, 2], dtype='float32'), name = 'W1')
        self.b1 = tf.Variable(tf.constant(0, shape=[2], dtype='float32'), name = 'b1')
        
        #self.W1_sigma = tf.Variable(tf.constant(0, shape=[self.num_inputs, 1], dtype='float32'), name = 'W1_sigma')
        #self.b1_sigma = tf.Variable(tf.constant(0, shape=[1], dtype='float32'), name = 'b1_sigma')
        
               
        #self.W2_mu = tf.Variable(tf.constant(0, shape=[n1, 1], dtype='float32'), name = 'W2_mu')
        #self.b2_mu = tf.Variable(tf.constant(0, shape=[1], dtype='float32'), name = 'b2_mu')        
        
        #self.W2_sigma = tf.Variable(tf.constant(0, shape=[n1, 1], dtype='float32'), name = 'W2_sigma')
        #self.b2_sigma = tf.Variable(tf.constant(0, shape=[1], dtype='float32'), name = 'b2_sigma')        
        
        
        net = tf.add(tf.matmul(self.Input, self.W1), self.b1)        
        net_mu = tf.gather_nd(net, [[0,0]])
        net_sigma = tf.gather_nd(net, [[0,1]])
        
        self.mu = net_mu
        self.sigma = tf.nn.softplus(net_sigma) + 1e-4
               
        #self.sample_op = tf.random.normal(
        #      shape = (1,1),
        #      mean=tf.reshape(self.mu, shape=(1,)),    
        #      stddev=tf.reshape(self.sigma, shape=(1,))
        #)
        
        dist = tf.distributions.Normal(loc=tf.reshape(self.mu, shape=(1,)), scale=tf.reshape(self.sigma, shape=(1,)))
        log_probs = dist.log_prob(self.selected_action)
                
        self.cost = -tf.reduce_mean(tf.multiply(self.advantage, log_probs))
        
        self.optimize_step = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.cost)
        self.init_op = tf.global_variables_initializer()
    
    def initialize_variables_and_sess(self):
        self.sess = tf.Session()
        self.sess.run(self.init_op)
        
    
    def predict_mu_sigma(self, input_features):
        sigma, mu = self.sess.run([self.sigma, self.mu], feed_dict={self.Input: input_features})
        return mu, sigma
        
    
    def update(self, input_features, advantage, selected_action):
        self.sess.run(self.optimize_step, feed_dict = {self.Input: input_features, self.advantage: advantage, self.selected_action: selected_action})
    
    def close_sess(self):
        self.sess.close()

if __name__ == '__main__':
    agent = Agent()
    agent.train_using_actor_critic(N=250,M=10,eps=1)
    agent.test(N=100)
    agent.value_model.close_sess()
    agent.policy_model.close_sess()
