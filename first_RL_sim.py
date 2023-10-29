# pip install stable-baselines3[extra]
# Python version 3.9.18


import gym 
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

    def __init__(self,clerk,waitingline):
        super().__init__()
        self.action = 0
        self.clerk = clerk
        self.waitingline = waitingline

    def process(self):
        global action_Cust
        #print("Created",action_Cust)
        while True:
            Customer(clerk = self.clerk,waiting_line= self.waitingline)
            self.hold(sim.Uniform(5, action_Cust).sample())

class Customer(sim.Component):
    def __init__(self,clerk,waiting_line):
        super().__init__()
        self.action = 0
        self.clerk = clerk
        self.waiting_line = waiting_line
    def process(self):
        self.enter(self.waiting_line)
        if self.clerk.ispassive():
            self.clerk.activate()
        self.passivate()


class Clerk(sim.Component):
    def __init__(self,waiting_line):
        super().__init__()
        self.action = 0
        self.waitingline = waiting_line


    def process(self):
        while True:
            while len(self.waitingline) == 0:
                self.passivate(30)
            self.customer = self.waitingline.pop()
            self.hold(30)
            self.customer.activate()

#global env_Sim 
env_Sim = sim.Environment(trace=False)
def run_sim(random):
    
    global env_Sim 
    env_Sim = sim.Environment(trace=False)
    waitingline = sim.Queue("waitingline88")
    clerk = Clerk(waiting_line=waitingline)
    CustomerGenerator(clerk = clerk,waitingline = waitingline)
    
    global action_Cust
 
    env_Sim.run(till=1440)
    #print(np.random.rand(1))

    return (waitingline.length() + random)

#Create 2a truck enviroment that the model is going to perform in
class TruckEnv(Env):
    def __init__(self):
        self.action_space = Box(low = -3, high = 3, shape = (1,), dtype = np.float64)
        self.observation_space = Box(low = -3, high = 250, shape = (1,), dtype = np.float64)
        self.state = 0
        self.done = False
        self.running = False
        self.wait_Timep = np.array([0],dtype = np.float32)
        self.bad_counter = 0
        self.loop = 2 
        self.state = 20 + random.randint(-3,3)
        #self.wait_Timep.astype(np.float32)


    def step(self,action):   
        
        global action_Cust

        #print( action_Cust)
        

        action_Cust += action[0]
        if action_Cust < 5:
            action_Cust = 5
        wait_time = run_sim(0)
        self.state[0] = wait_time
        self.wait_Timep[0] = np.float32(wait_time)
        self.loop +=1
        #print(self.wait_Timep[0])       
        info = {}
        if wait_time >=90 and wait_time <110:
            reward = 1
            self.bad_counter = 0
            #print("Ok")
        elif wait_time <30 or wait_time >200:
            reward = -2
        else:
            reward = -1
            self.bad_counter +=1
            #print("not_Ok")

        if self.loop == 8:
            done = True
            #print("Loop Count", self.loop)
        else:
            #print("False")
            done = False
       #print(wait_time)
        return np.float32(self.state),reward, done,False, info
    
    def render(self):
        pass

    def reset(self,seed =None):
        global waitingline
        #print("Reset")
        #waitingline = sim.Queue("waitingline88")
        info= {}
        self.state = np.array([200 + random.randint(-3,3) ]).astype(np.float32)
        #print(self.state)
        obs = np.array
        self.loop = 0
        return np.float32(self.state), info
    
env = TruckEnv()   
from stable_baselines3.common.env_checker import check_env
check_env(env, warn=True)
env.close()

log_path = os.path.join('.','logs')
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps= 20000, progress_bar = True)

