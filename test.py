import numpy as np

# Given parameters
lambda_rate = 40 / 60  # arrival rate in students per minute
mu_rate = 50 / 60  # service rate in students per minute

# Simulation parameters
num_students = 1000000  # number of students to simulate

# Generate inter-arrival times for the students (Poisson process)
inter_arrival_times = np.random.exponential(1/lambda_rate, num_students)

# Generate service times for the students (exponential distribution)
service_times = np.random.exponential(1/mu_rate, num_students)

# Initialize lists to store arrival times and departure times
arrival_times = [0] * num_students
departure_times = [0] * num_students

# Calculate arrival times and departure times
for i in range(num_students):
    if i == 0:
        arrival_times[i] = inter_arrival_times[i]
    else:
        arrival_times[i] = arrival_times[i-1] + inter_arrival_times[i]
    
    if i == 0 or departure_times[i-1] <= arrival_times[i]:
        departure_times[i] = arrival_times[i] + service_times[i]
    else:
        departure_times[i] = departure_times[i-1] + service_times[i]

# Calculate waiting times before service
waiting_times_before_service = [departure_times[i] - arrival_times[i] - service_times[i] for i in range(num_students)]

# Calculate average waiting time before service
average_waiting_time_before_service = np.mean(waiting_times_before_service)

print(f"Average waiting time before service: {average_waiting_time_before_service} minutes")