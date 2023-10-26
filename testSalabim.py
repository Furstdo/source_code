# pip install stable-baselines3[extra]
# Python version 3.9.18
import gym
from gym import Env
from gym.spaces import Discrete,Box,Dict,Tuple,MultiBinary,MultiDiscrete

import numpy as np
import random
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

import simpy
import random
import statistics
import numpy as np
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
def run_sim():
    global env_Sim 
    env_Sim = sim.Environment(trace=False)
    waitingline = sim.Queue("waitingline88")
    clerk = Clerk(waiting_line=waitingline)
    CustomerGenerator(clerk = clerk,waitingline = waitingline)
    
    global action_Cust
 
    env_Sim.run(till=1440)
    return (waitingline.length())

action_Cust =5
run_sim()
action_Cust = 9
run_sim()

#Create a truck enviroment that the model is going to perform in
class TruckEnv(Env):
    def __init__(self):
        self.action_space = Box(low = np.array([5]), high = np.array([20]))
        self.observation_space = Box(low = np.array([0]), high = np.array([1000]))
        self.state = 0
        self.done = False
        self.running = False
        #self.env_sim = env = sim.Environment(trace=False)    


    def step(self,action):   
        global action_Cust
        print(action[0])
        action_Cust = action[0]
        wait_time = run_sim()
        done = True         
        info = {}
        if wait_time >90 and wait_time <110:
            reward = 1
        else:
            reward = -1

        return wait_time, reward, done, info
    
    def render(self):
        pass

    def reset(self):
        global waitingline
        waitingline = sim.Queue("waitingline88")

    
    
    
    
env = TruckEnv()
log_path = os.path.join('.','logs')
model = PPO('MlpPolicy', env, verbose = 1, tensorboard_log = log_path)

model.learn(total_timesteps= 200)
