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
    def __init__(self,waiting_room,env,clerks,wait_times):
        super().__init__()
        self.waiting_room = waiting_room
        self.env = env
        self.clerks = clerks
        self.debug = False
        self.wait_times = wait_times

    def process(self):
        while True:
            cust =Customer(waiting_room= self.waiting_room,env=self.env,stations=self.clerks,wait_times = self.wait_times)
            cust.creation_time = self.env.now()
            self.hold(sim.Uniform(30, 35).sample())
            self.debug = True

class Charging_Station(sim.Component):
    def __init__(self,waiting_room,env):
        super().__init__()
        self.waiting_room = waiting_room
        self.vehicle = 0
        self.env =  env

    def process(self):
        while True:
            #Continu looping until a vehicle shows up in the waiting line
            while len(self.waiting_room) == 0:
                self.passivate()
            self.vehicle = self.waiting_room.pop()
            self.charge_car()
            

    #This method charges car and stops when the car has been charged
    def charge_car(self):
        loop = 0
        while self.vehicle.battery_charge < 100:
            self.hold(1)
            self.vehicle.battery_charge +=1
            loop +=1
        #Calculate the time that the complete charging procedure took
        self.vehicle.wait_times.append(self.env.now() - self.vehicle.creation_time)
        return loop


class Customer(sim.Component):
    def __init__(self,waiting_room,env,stations,wait_times):
        super().__init__()
        self.waiting_room = waiting_room
        self.env = env
        self.stations = stations
        self.battery_charge = int(random.uniform(20, 80))
        self.creation_time = 0
        self.wait_times = wait_times

    def process(self):        
        
        #Put the vehicle in the waiting room
        self.enter(self.waiting_room)
        #Check if there is a station that is passive
        for station in self.stations:
            if station.ispassive():
                station.activate()
                break  # activate at most one clerk
        self.passivate()

generator_ok = False

#This function runs the simmulation
def run_sim(amount_clerks):
    #Create varaibles for monitoring
    wait_Times = []
    #Create the objects that make up the simmulation
    env_Sim = sim.Environment(trace=False)
    waiting_room = sim.Queue("waitingline88")
    clerks = [Charging_Station(waiting_room=waiting_room,env=env_Sim) for _ in range(amount_clerks)]
    generator = CustomerGenerator(waiting_room= waiting_room,env=env_Sim,clerks=clerks,wait_times = wait_Times)
    #Start the simmulation
    env_Sim.run(till=1440)
    #Delete the objects from the memory
    del(env_Sim)
    del(waiting_room)
    for clerk in clerks:
        del(clerk)
    del(generator)

    #Get the output of the simmulation
    avg = sum(wait_Times)/len(wait_Times)
    min_o = min(wait_Times)
    max_o = max(wait_Times)

    return int(avg),int(min_o),int(max_o)

print(run_sim(100))

#Create a truck enviroment that the model is going to perform in
class TruckEnv(Env):
    def __init__(self):
        self.action_space = Discrete(100)
        #self.action_space = Box(low = -0, high = 10, shape = (1,), dtype = int)
        self.observation_space = Box(low = -3, high = 250, shape = (1,), dtype = np.float64)
        self.state = 0
        self.done = False
        self.running = False
        #self.env_sim = env = sim.Environment(trace=False)    

    def step(self,action):           
        print(action)
        wait_time = run_sim(action)
        self.state = 0
        reward = 1 
        done = True         
        info = {}
        if wait_time >70 and wait_time <80:
            reward = 1

        else:
            reward = -1

        return np.float32(self.state),reward, done,False, info
    
    def render(self):
        pass

    def reset(self,seed =None):
        self.state = 100
        info = {}
        return np.float32(self.state), info   

  
#env = TruckEnv()
log_path = os.path.join('.','logs')
#model = PPO('MlpPolicy', env, verbose = 1, tensorboard_log = log_path)

#model.learn(total_timesteps= 10000,progress_bar= True)




