# pip install stable-baselines3[extra]
# Python version 3.9.18
import gym 
import copy
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 
import numpy as np
import random
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

import matplotlib.pyplot as plt

# Bank, 1 clerk.py
import salabim as sim


action_Cust = 5

class CustomerGenerator(sim.Component):
    def __init__(self,waiting_room,env,clerks):
        super().__init__()
        self.waiting_room = waiting_room
        self.env = env
        self.clerks = clerks

    def process(self):
        while True:
            Customer(waiting_room= self.waiting_room,env=self.env,clerks=self.clerks)
            self.hold(sim.Uniform(5, 15).sample())

class Clerk(sim.Component):
    def __init__(self,waiting_room):
        super().__init__()
        self.waiting_room = waiting_room

    def process(self):
        while True:
            while len(self.waiting_room) == 0:
                self.passivate()
            self.customer = self.waiting_room.pop()
            self.hold(300)
            self.customer.activate()


class Customer(sim.Component):
    def __init__(self,waiting_room,env,clerks):
        super().__init__()
        self.waiting_room = waiting_room
        self.env = env
        self.clerks = clerks

    def process(self):
        self.hold(50)
        self.enter(self.waiting_room)
        for clerk in self.clerks:
            if clerk.ispassive():
                clerk.activate()
                break  # activate at most one clerk
        self.passivate()

generator_ok = False
#global env_Sim 

def run_sim(amount_clerks):
    env_Sim = sim.Environment(trace=False)
    waiting_room = sim.Queue("waitingline88")
    clerks = [Clerk(waiting_room=waiting_room) for _ in range(amount_clerks)]
    CustomerGenerator(waiting_room= waiting_room,env=env_Sim,clerks=clerks)
    env_Sim.run(till=1440)
    length = copy.deepcopy(waiting_room.length())
    del(env_Sim)
    del(waiting_room)
    for clerk in clerks:
        del(clerk)
    return length



#Create a truck enviroment that the model is going to perform in
class TruckEnv(Env):
    def __init__(self):
        self.action_space = Discrete(10)
        #self.action_space = Box(low = -0, high = 10, shape = (1,), dtype = int)
        self.observation_space = Box(low = -3, high = 250, shape = (1,), dtype = np.float64)
        self.state = 0
        self.done = False
        self.running = False
        #self.env_sim = env = sim.Environment(trace=False)    


    def step(self,action):   
        run_sim(action)
        wait_time = 0
        self.state = 0
        reward = 1 
        done = True         
        info = {}
        if wait_time >90 and wait_time <110:
            reward = 1
            print("Ok")
        else:
            reward = -1

        return np.float32(self.state),reward, done,False, info
    
    def render(self):
        pass

    def reset(self,seed =None):
        self.state = 100
        info = {}
        return np.float32(self.state), info   
    
    
env = TruckEnv()
log_path = os.path.join('.','logs')
model = PPO('MlpPolicy', env, verbose = 1, tensorboard_log = log_path)

model.learn(total_timesteps= 20000,progress_bar= True)
