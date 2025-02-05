# pip install stable-baselines3[extra]
# Python version 3.9.18
import gym 
import copy
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 
import numpy as np
import random
import os
import time
from scipy.stats import expon
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from dataclasses import dataclass
import matplotlib.pyplot as plt
import salabim as sim
import gc

#-------------------------------------------------------------------------------------------
def limit(lower,value,max):
    if value < lower:
        return lower
    elif value > max:
        return max
    else:
        return value
    
#-------------------------------------------------------------------------------------------
#Struct that holds the information regarding a truck
@dataclass
class Truck:
    Battery:                np.int16
    Arrival_Time:           np.int16
    total_time:             np.int16
    total_wait_Time:        np.int16

#Trcuk that hold charging station information
@dataclass
class Consumption:
    Power_Consumption:      np.real#Current consumption
    Max_Power_Consumption:  np.real#Max consumption from the 
    Max_Power_Reqeust:      np.real#Max power the charging station is able to get

#-------------------------------------------------------------------------------------------
#Class that prepares a car arrival set
class Prepare():
    def __init__(self,total_time):
        #Create an empty list where we can store the truck scedual
        self.trucks = []
        self.total_time = total_time
        random.seed(time.time())

    def prepare_data(self,spread_type):
        self.trucks = []
        time = 0
        first = False   
        #Loop until a day is finished
        while time <self.total_time:                 
            if spread_type == 1:     
                #Create a new data object
                Truck_Data = Truck(Battery= sim.Uniform(20, 80).sample(),Arrival_Time=time,total_time=0)  
            elif spread_type == 2:                            
                Truck_Data = Truck(Battery= sim.Uniform(40).sample(),Arrival_Time=time,total_time=0,total_wait_Time=0)   
            elif spread_type == 3: 
                arrival, service_time = self.poison()
                service_invert = 100 - service_time
                Truck_Data = Truck(Battery= service_invert,Arrival_Time=time,total_time=0,total_wait_Time=0)   
            #Append the data to the list
            self.trucks.append(Truck_Data)   
            #Determine the new arrival time
            if first == False: 
                time += arrival
            else:
                first = True

    def poison(self):
        # Given parameters
        lambda_rate = 100 / 60  # arrival rate in students per minute
        mu_rate = 50 / 60  # service rate in students per minute
        # Generate inter-arrival times for the students (Poisson process)
        arrival_time = np.random.exponential(1/lambda_rate, 1)
        # Generate service times for the students (exponential distribution)
        service_times = np.random.exponential(1/mu_rate, 1)
        return 1,60
        #return arrival_time[0],service_times[0]
    

#-------------------------------------------------------------------------------------------
class CustomerGenerator(sim.Component):
    def __init__(self,waiting_room,env,clerks,wait_times,time_before_service,shedual):
        super().__init__()
        self.waiting_room = waiting_room
        self.env = env
        self.clerks = clerks
        self.debug = False
        self.wait_times = wait_times
        self.time_before_service = time_before_service
        self.shedual = shedual

    def process(self):
        previous_arrival = 0
        while True:            
            #Check if there ia an truck left in the list
            if len(self.shedual) > 0:
                #Get the next truck out of the list
                truck = self.shedual.pop(0)
            else:
                #print("Break")
                #Break when there are no more trucks to create
                break
            #Create a truck object
            cust =Customer(waiting_room= self.waiting_room,env=self.env,stations=self.clerks,wait_times = self.wait_times, time_before_service = self.time_before_service,battery_charge=truck.Battery)
            cust.creation_time = self.env.now()
            #Hold the simmulation until the next truck is sheduald
            self.hold(truck.Arrival_Time - previous_arrival)
            #Set the previous time
            previous_arrival = truck.Arrival_Time


class Charging_Station(sim.Component):
    def __init__(self,waiting_room,env,power_supply,max_power_delivery):
        super().__init__()
        random.seed(time.time())
        self.waiting_room = waiting_room
        self.vehicle = 0
        self.power_supply = power_supply
        self.env =  env
        self.max_power_delivery = max_power_delivery
        self.power_consumption = Consumption(0,0,0)
        #Append the power consumption to the consumtion list
        self.power_supply.power_used_list.append(self.power_consumption)
        self.power_consumption.Max_Power_Reqeust = self.max_power_delivery

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
        add_Charge = 0
        self.vehicle.wait_times.append(self.env.now() - self.vehicle.creation_time)
        
        while self.vehicle.battery_charge < 100:
            #Determine the max power delivery that the charging station is able to give
            #max_power = limit(0,self.max_power_delivery,self.power_supply.)
            max_power = 0
            
            if self.vehicle.battery_charge < 100 - self.max_power_delivery:
                add_Charge = self.power_consumption.Max_Power_Consumption
                #print(add_Charge)
                #add_Charge = limit(0,self.max_power_delivery, 100 - limit(0,self.vehicle.battery_charge,)
            else:
                add_Charge = self.power_consumption.Max_Power_Consumption
            print("add_Charge",add_Charge)
            #Note to the power supply much power is being used from it
            self.power_consumption.Power_Consumption = add_Charge            
            #Hold the simulation for 1 minute
            self.hold(1)
            self.vehicle.battery_charge +=add_Charge
            #print(self.vehicle.battery_charge)
            loop +=1
        #Calculate the time that the complete charging procedure took
        return loop

    #This method calculates the maximum amount of charge the charging pole is allowed to give
    def max_power_consumption(self):
        #Calculate the total amount of power already used by the charging stations
        power_used = 0
        for i in self.power_supply.power_used:
            power_used += i
       

class Customer(sim.Component):
    def __init__(self,waiting_room,env,stations,wait_times,time_before_service,battery_charge):
        super().__init__()
        self.waiting_room = waiting_room
        self.env = env
        self.stations = stations
        self.battery_charge = battery_charge
        self.creation_time = 0
        self.wait_times = wait_times
        self.time_before_service = time_before_service

    def process(self):        
        #Put the vehicle in the waiting room
        self.enter(self.waiting_room)
        #print(len(self.waiting_room))
        #Check if there is a station that is passive
        for station in self.stations:
            if station.ispassive():
                station.activate()
                break  # activate at most one clerk
        self.passivate()

#This class resables the general power supply that the chraging stations are coupled to
class Power_supply(sim.Component):
    def __init__(self,env,max_power_from_Grid):
        super().__init__()
        self.max_power_from_grid = max_power_from_Grid
        self.power_used_list = []
        self.distribution_rl = []
        self.power_used = 0
        self.env =  env
        self.strategy = 0
        self.max_reached = False

    def process(self):
        #Calculate the amount of energy that is currently being used
        while True:
            total = 0
            #Select the charging strategy
            if self.strategy == 0:
                self.__distribute_power_simple()
            elif self.strategy == 1:
                self.__disrtibute_power_share_equal()
            elif self.strategy == 2:
                self.__distribute_power_rl(rl_distribution=self.distribution_rl)
            #Check if the list has members
            if len(self.power_used_list) != 0:
                #Loop through all the charging stations
                for i in self.power_used_list:
                    total += i.Power_Consumption
                self.hold(1)
                #print(total)
            else:
                pass

    def __distribute_power_simple(self):#This method resembles the simplest distribution (give max until it is out)
        #Loop through all the power cinsumers
        total_distributed = 0
        for i in self.power_used_list:
            #Calculate the max distribution left
            max_allowd = limit(0,self.max_power_from_grid - total_distributed,self.max_power_from_grid)
            #print("Max_Allowed",max_allowd)
            i.Max_Power_Consumption = limit(0,i.Max_Power_Reqeust,max_allowd)
            #if i.Max_Power_Consumption == 0:
                #print("No power_To_Pole")
            total_distributed += i. Max_Power_Consumption
                                                
    def __disrtibute_power_share_equal(self):#This method resables a equal share to all the charging stations
        #Loop through all the power consumers
        total_distributed = 0
        if len(self.power_used_list) != 0:
            available_per_station = self.max_power_from_grid / len(self.power_used_list)
            for i in self.power_used_list:#Calculate the total amount
                #Give the allowed power to the stations
                i.Max_Power_Consumption = limit(0,i.Max_Power_Reqeust,available_per_station)

    def __distribute_power_rl(self,rl_distribution):#This method is used to distribute the power with the help of reinforcemnt learning
        total_distributed = 0
        counter = 0
        if len(self.power_used_list) != 0:
            for i in self.power_used_list:
                max_allowd = limit(0,self.max_power_from_grid - total_distributed,self.max_power_from_grid)
                max_allowd = limit(0,max_allowd,i.Max_Power_Reqeust)
                max_allowd = limit(0,max_allowd,limit(0,self.max_power_from_grid - total_distributed,self.max_power_from_grid - total_distributed))
                #Note to the system when the maximum energy consumption is reached
                if max_allowd == 0:
                    self.max_reached = True
                else:
                    self.max_reached = False
                #Insert the max power consumption from the reinforcement learning model into 
                i.Max_Power_Consumption = limit(0,rl_distribution[counter],max_allowd)       
                total_distributed += i. Max_Power_Consumption
                counter += 1

#-------------------------------------------------------------------------------------------
class sim_manager():
    def __init__(self,Charging_Stations,total_time):
        self.shedual = Prepare(total_time=total_time)
        self.Charging_stations = Charging_Stations
        #Prepare the truck data
        self.shedual.prepare_data(spread_type=3)
        self.total_time = total_time
    #This function runs the simmulation
    def run_sim(self):
        #Create varaibles for monitoring
        wait_Times = []
        time_before_service = []
        #Prepare the truck data

        #Create the objects that make up the simmulation
        env_Sim = sim.Environment(trace=False,)
        env_Sim.Monitor('.',stats_only= True)
        waiting_room = sim.Queue("waitingline88")
        Power_supply_o = Power_supply(env =env_Sim,max_power_from_Grid= 200)
        Power_supply_o.distribution_rl = [10,20,20]
        Power_supply_o.strategy = 2
        stations = [Charging_Station(waiting_room=waiting_room,env=env_Sim,power_supply=Power_supply_o,max_power_delivery=20) for _ in range(self.Charging_stations)]
        generator = CustomerGenerator(waiting_room= waiting_room,env=env_Sim,clerks=stations,wait_times = wait_Times,time_before_service=time_before_service,shedual= self.shedual.trucks )

        #Start the simmulation
        env_Sim.run(till=self.total_time)
        #Delete the objects from the memory
        del(env_Sim)
        del(waiting_room)
        for station in stations:
            del(station)
        del(generator)


        #Get the output of the simmulation
        avg = sum(wait_Times)/len(wait_Times)
        min_o = min(wait_Times)
        max_o = max(wait_Times)

        return avg,int(min_o),int(max_o)

    def reset_shedual(self):
        self.shedual.prepare_data(spread_type= 3)

count = 0 

for i in range(1):
    if count >1000:
        print("1000 passed")
        count = 0
    count += 1
    sim_m = sim_manager(3,1400)
    print(sim_m.run_sim())

#-------------------------------------------------------------------------------------------
#Create a truck enviroment that the model is going to perform in
class TruckEnv(Env):
    def __init__(self):
        self.action_space = Discrete(100)
        #self.action_space = Box(low = -0, high = 10, shape = (1,), dtype = int)
        self.observation_space = Box(low = -3, high = 250, shape = (1,), dtype = np.float64)
        self.state = 0
        self.done = False
        self.running = False
        self.sim_env = sim_manager(3,10000)
        #self.env_sim = env = sim.Environment(trace=False)    

    def step(self,action):           
        print(action)
        wait_time = self.sim_env.run_sim(action)
        
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
        super().reset(seed=seed) 
        #Reset the simmulation enviroment
        self.sim_env.reset_shedual()
        #Get a local copy of the schedule
        schedule = self.sim_env.shedual.trucks

        battery = []
        arriaval_time =[]
        #Extract the data from the schedule
        for i in schedule:
            battery.append(i.Battery)
            arriaval_time.append(i.Arrival_Time)
        print(arriaval_time)
        battery_np = np.array(battery)
        arriaval_time_np = np.array(arriaval_time)   
        #Create a 
        self.state = 100
        info = {}
        return np.float32(self.state), info   

  
env = TruckEnv()
#env.reset()
log_path = os.path.join('.','logs')
#model = PPO('MlpPolicy', env, verbose = 1, tensorboard_log = log_path)

#model.learn(total_timesteps= 10000,progress_bar= True)




