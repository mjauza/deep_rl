# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 21:44:59 2019

@author: David
"""

import numpy as np
import gym
from gym import wrappers
import random
import matplotlib.pyplot as plt

class Model():
    
    def __init__(self, lr,list_of_bounds, num_splits=9, num_actions = 2):
        self.lr = lr
        self.num_splits = num_splits
        self.list_of_bounds = list_of_bounds
        self.num_actions = num_actions
        self.define_boxes()
    
    def define_boxes(self):
        boxes  = []
        for bounds in self.list_of_bounds:
            mini = bounds[0]
            maxi = bounds[1]
            splits = np.linspace(mini,maxi,self.num_splits)
            splits = np.insert(splits,0,-np.inf)
            splits = np.append(splits,np.inf)
            boxes.append(splits.tolist())
        self.boxes = np.array(boxes)
    
    def get_indexes_from_state(self,state):
        indexes = []
        for i,s in enumerate(state):
            bounds = self.boxes[i]
            idx = self.get_index_in_bounds(s,bounds)
            indexes.append(idx)
        return tuple(indexes)
    
    def get_Q_indexes_from_state_action(self,state,action):
        assert action in list(range(self.num_actions)), 'action must be in action space'
        state_indexes = self.get_indexes_from_state(state)
        state_list = list(state_indexes)
        state_list.append(action)
        return tuple(state_list)
    
    def get_index_in_bounds(self,x,bounds):
        bounds = np.array(bounds)
        assert np.all(bounds == np.sort(bounds)) , 'bounds should be sorted'
        idx = np.where(x > bounds)[0][-1]
        return idx
    
    def initialize_shape_of_Q(self):        
        shape = [int(self.boxes.shape[1] - 1)] * self.boxes.shape[0]
        shape.append(self.num_actions)
        self.Q_shape = tuple(shape)        
    
    def initialize_Q(self):
        self.initialize_shape_of_Q()
        self.Q = np.zeros(shape=self.Q_shape)
    
    def update_Q(self,state,action,target):
        #get index of Q from state action
        idx = self.get_Q_indexes_from_state_action(state,action)
        self.Q[idx] = self.Q[idx] + self.lr*(target - self.Q[idx])
    
    def get_Q_from_state(self,state):
        state_indexes = self.get_indexes_from_state(state)
        return self.Q[state_indexes].copy()
    
    def get_Q_from_state_action(self,state,action):
        idx = self.get_Q_indexes_from_state_action(state,action)
        return self.Q[idx]
        

class Agent():
    def __init__(self,gamma):
        self.env = gym.make('CartPole-v0')
        self.env_record = wrappers.Monitor(self.env, 'video_binned_state_space')
        self.list_of_bounds = [[-2.4,2.4], [-2,2], [-0.4,0.4], [-3.5,3.5]]
        self.num_splits = 9
        self.lr = 0.001
        self.num_actions = 2
        self.gamma = gamma
        self.Q_model = Model(lr = self.lr, list_of_bounds=self.list_of_bounds, num_splits=self.num_splits,num_actions=self.num_actions)
        self.Q_model.initialize_Q()
        
    def get_acion_eps_greedy(self,eps,state):
        u = np.random.uniform()
        if u < eps:
            selected_action = random.randint(0,self.num_actions - 1)
            #print('random')
        else:
            q_values = self.Q_model.get_Q_from_state(state)
            selected_action = np.argmax(q_values)
            #print('greedy')
        return selected_action            
    
    def play_episode_q_learning(self,eps):
        #pick initial state
        s = self.env.reset()
        done = False
        acc_r = 0
        while not done:
            a = self.get_acion_eps_greedy(eps=eps,state=s)
            #print('a = ', a)
            s_prime, r, done, _ = self.env.step(a)
            if done:
                target = r
            else:
                Q_s_prime = self.Q_model.get_Q_from_state(s_prime)
                target = r + self.gamma * np.max(Q_s_prime)
            
            #update Q
            self.Q_model.update_Q(s,a,target)
            s = s_prime
            acc_r += r
        return acc_r
        
    
    
    def train_using_q_learning(self,eps,N,M):
        avg_rewards = []
        for i in range(N):
            rewards = []
            for j in range(M):
                reward = self.play_episode_q_learning(eps = eps)
                rewards.append(reward)
            avg_rewards.append(np.mean(np.array(rewards)))
            print('i =',i)
        fig = plt.figure()
        plt.plot(np.array(avg_rewards))
        plt.savefig('q_learning.png')
        fig.show()
    
    def test_strategy_and_record(self,N=100):        
        rewards = []
        for i in range(N):
            s = self.env_record.reset()
            done = False
            episode_r = 0
            while not done:
                a = self.get_acion_eps_greedy(eps=0,state=s)
                if a not in [0,1]:
                    raise Exception('wrong action')
                s, r, done, _ = self.env_record.step(a)
                episode_r += r
            rewards.append(np.mean(np.array(episode_r)))
        fig = plt.figure()
        plt.plot(rewards)
        plt.savefig('test_rewards.png')
        fig.show()
                
       
    
    
agent = Agent(gamma = 1)
agent.train_using_q_learning(eps=0.5,N=1000, M=100)
agent.test_strategy_and_record()


            
            
        
        