# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 20:30:57 2019

@author: David
"""

import gym
import numpy as np
#from sklearn.linear_model import SGDRegressor
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
        
        
    def play_episode_q_learning(self,eps):
        #initialize env
        s = self.env.reset()
        done = False
        acc_r = 0
        while not done:
            #pick action eps gredy
            a = self.pick_action_eps_greedy(eps=eps,state=s)
            
            #move
            s_prime, r, done, _ = self.env.step(a)
            
            #create target
            if done:
                target = r
            else:
                s_prime_reshaped = np.reshape(s_prime, (1,-1))
                input_features_s_prime = self.feat_transform.transform(s_prime_reshaped)
                Q_s_prime = self.Q_model.predict(input_features_s_prime)[0]
                #print('shape Q_s_prime = ',Q_s_prime.shape)
                target = r + self.gamma * np.max(Q_s_prime)
                
            
            #get full target
            full_target = self.get_full_target(state=s,action=a,target=target)
            
            #update model
            s_reshaped = np.reshape(s, (1,-1))
            input_features_s = self.feat_transform.transform(s_reshaped)
            self.Q_model.update(input_features=input_features_s, target = full_target)
            
            s = s_prime
            acc_r += r
        return acc_r
    
    def train_q_learning(self,N,M,eps):
        rewards = []
        for i in range(N):
            rew = []
            for j in range(M):
                r = self.play_episode_q_learning(eps=eps)
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
            a = self.pick_action_eps_greedy(eps=eps,state=s)
            s_prime, r, done, _ = self.env_record.step(a)
            s = s_prime
            acc_r += r
        return acc_r
            
    
    def test(self,N):
        self.env_record = wrappers.Monitor(self.env, 'mountain_car_rbf')
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
        
        n1 = 200
        W1 = tf.Variable(tf.random_normal([self.num_input_features, n1], stddev=1), name='W1')
        b1 = tf.Variable(tf.random_normal([n1]), name='b1')
        
        n2 = 100
        W2 = tf.Variable(tf.random_normal([n1, n2], stddev=1), name='W2')
        b2 = tf.Variable(tf.random_normal([n2]), name='b2')
        
        W3 = tf.Variable(tf.random_normal([n2, self.num_actions], stddev=1), name='W3')
        b3 = tf.Variable(tf.random_normal([self.num_actions]), name='b3')
        
        net = tf.add(tf.matmul(Input, W1),b1)
        net = tf.nn.relu(net)
        net = tf.add(tf.matmul(net, W2), b2)
        net = tf.nn.relu(net)
        output = tf.add(tf.matmul(net, W3), b3)
        
        loss = tf.reduce_mean(tf.squared_difference(output, Label))
        optimize_step = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(loss)
        init_op = tf.global_variables_initializer()
        
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
    
    def close_sess(self):
        self.sess.close()
    
    
if __name__ == '__main__':
    agent = Agent()
    agent.train_q_learning(N=1000,M=10,eps=1)
    agent.test(N=100)
    agent.Q_model.close_sess()
        
        