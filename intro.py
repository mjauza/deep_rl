# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 20:36:26 2019

@author: David
"""

#import gym
#get enviroment
#env = gym.make('CartPole-v0')

#reset enviroment, returns a start state
#env.reset()

#box
#box = env.observation_space

#action sapece
#env.action_space

#get observation, reward, done flag, info
#obsertvation, reward, done, info = env.step(action)

#finish an episode example
'''
done = False
while not done:
    observation, reward, done, _ = env.step(env.action_space.sample())
'''

import numpy as np
import gym
from gym import wrappers

def get_action(s,w):
    return 1 if s.dot(w) > 0 else 0

def play_episode_with_w(w,env):
    done = False
    s = env.reset()
    while not done:
        a = get_action(s,w)
        s, r, done, _ = env.step(a)
        

def play_one_episode(env):
    t = 0
    new_weights = np.random.normal(size=(4,))    
    done = False
    s = env.reset()    
    duration = 0
    while not done and t < 1000:
        a = get_action(s,new_weights)
        s, r, done, _ = env.step(a)        
        duration += 1
        t += 1    
    return duration, new_weights

def train(N, env):    
    w = None
    max_duration = -1
    for i in range(N):
        duration, new_w = play_one_episode(env)
        if duration > max_duration:
            max_duration = duration
            w = new_w
        print('i = ',i)
    print('max_duration = ', max_duration)
    return w

def train_and_play(N):
    env = gym.make('CartPole-v0')        
    w = train(N=N, env=env)    
    env = wrappers.Monitor(env, 'video')
    play_episode_with_w(w,env)
    

#train_and_play(N=10000)
        
        