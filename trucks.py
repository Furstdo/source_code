import simpy
import random
import statistics
import numpy as np
import matplotlib.pyplot as plt

battery_trend = []
wait_times = []
time_stamp = []
print("fre")
#Create an class that mimics an charging station
class Charging_station():
    #Init method for the charging station class
    def __init__(self,env,num_charging_stations):
        self.env = env
        self.charge_station = simpy.Resource(env,num_charging_stations)
        self.battery_amount =150000
        self.amount = 2
        self.battery_drained = False
        self.solar_power = 0

    #This method controls the solar panel 
    def solar_panel(self):
        while True:
            yield self.env.timeout(0.016)
            #This method controls the solar panel
            self.battery_amount += self.solar_power
            self.solar_power += 0.000006

    def trend_battery(self):
        time = 0
        while True:
            #Trend the battery status every minute
            yield self.env.timeout(1)
            battery_trend.append(self.battery_amount)  
            time_stamp.append(time)
            time +=1


    #This method simmulates the charging of a vechicle
    def charge_vechicle(self, vehicle):
        # Set a random time for 
        if self.battery_amount >1:
            self.amount = 2
        else:
            if self.battery_drained == False:
                print("Battery Drained")
                self.battery_drained = True 
            self.amount = 1
        
        self.battery_amount -= 1
        yield self.env.timeout(0.016)        

class Truck():
    #Init the truck class
    def __init__(self,charge_left,number):
        self.charge = charge_left
        self.number = number

    def charge_truck(self,env,truck,charging_station):
        #Get the starting time
        begin_time = env.now
        #Charge the truck
        with charging_station.charge_station.request() as request:
            #Wait for an free charging station
            yield request
            #Start the charging
            while self.charge < 7200:
                yield env.process(charging_station.charge_vechicle(truck))
                self.charge += charging_station.amount
        #Determine the total time a truckdriver is needing for a charge
        wait_times.append(env.now -begin_time) #Deze moet wel aan blijven!!!!!!!



#This function runs the charging station
def run_station(env,num_chargers):
    truck = []
    number = 1
    #Create the simmulation enviroment
    charge_station = Charging_station(env,num_chargers )     
    #Set 1 car in the waiting line
    truck.append(Truck(50,1))
    #Add the solat panel to the enviroment\
    env.process(charge_station.solar_panel())
    env.process(truck[1 -1].charge_truck(env,truck,charge_station))
    env.process(charge_station.trend_battery())
    while True:
        yield env.timeout(5)
        truck.append(Truck(50,1))
        number += 1
        truck.append(Truck(50,number))
        env.process(truck[number -1].charge_truck(env,truck,charge_station))



def get_average_wait_time(arrival_times):
    average_wait = statistics.mean(wait_times)
    # Pretty print the results
    minutes, frac_minutes = divmod(average_wait, 2)
    seconds = frac_minutes * 60
    return round(minutes), round(seconds)

#The main method for the systme
def main(): 
    #Setup the main system
    random.seed(30)
    #Run the simmulation
    env = simpy.Environment()
    env.process(run_station(env,5))

    #Run the simmulation for 1 day
    env.run(until= 1440)
    #View the results
    mins, secs = get_average_wait_time(wait_times)
    print("Running simmulation...", 
          f"\nThe average wait time is {mins} minutes and {secs} seconds.")
    
    print()
    
main()

y = np.array(battery_trend)
x = np.array(time_stamp)
slope, intercept = np.polyfit(x, y, 1)
plt.plot(x, y)
plt.show()

