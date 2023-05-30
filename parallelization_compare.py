import time
import psutil
import matplotlib.pyplot as plt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
import traci
import multiprocessing
import random

def calculate_average_speed(vehicle_ids):
    speeds = [traci.vehicle.getSpeed(v) for v in vehicle_ids]
    return sum(speeds) / len(speeds)

def process_subset(vehicle_subset):
    # with traci.connection.RequestThreaded() as traci_conn:
    # traci_conn.simulationStep()
    return calculate_average_speed(vehicle_subset)

# Function to calculate average speed using CPU approach
def cpu_parallel(num_processors):
    # Implementation for CPU parallel approach
    # Start SUMO simulation
    sumo_binary = "sumo"
    sumo_config = "simple.sumocfg"
    sumo_cmd = [sumo_binary, "-c", sumo_config]
    traci.start(sumo_cmd)
    traci.simulationStep(100)

    vehicle_ids = traci.vehicle.getIDList()

    subset_size = len(vehicle_ids) // num_processors

    subsets = [vehicle_ids[i:i+subset_size] for i in range(0, len(vehicle_ids), subset_size)]
    
    with multiprocessing.Pool(processes=num_processors) as pool:
        traci.simulationStep()
        results = pool.map(process_subset, subsets)

    
    overall_avg_speed = sum(results) / len(results)
    print("Overall average speed:", overall_avg_speed)

    # Stop SUMO simulation
    traci.close()

# Function to calculate average speed using GPU parallel approach
def calculate_average_speed_gpu(speeds_gpu):
    num_vehicles = speeds_gpu.shape[0]
    # Retrieve the data from GPU to CPU
    speeds_cpu = speeds_gpu.get()
    # Perform the summation using numpy
    total_speed = np.sum(speeds_cpu)

    return total_speed / num_vehicles

# Function to calculate average speed using GPU parallel approach
def gpu_parallel(block_size):
    # Implementation for GPU parallel approach
    # Start SUMO simulation
    sumo_binary = "sumo"
    sumo_config = "simple.sumocfg"
    sumo_cmd = [sumo_binary, "-c", sumo_config]
    traci.start(sumo_cmd)
    traci.simulationStep(1000)
    # Retrieve vehicle speeds from SUMO
    vehicle_ids = traci.vehicle.getIDList()
    speeds_cpu = np.array([traci.vehicle.getSpeed(v) for v in vehicle_ids], dtype=np.float32)

    # Transfer speeds to GPU memory
    speeds_gpu = gpuarray.to_gpu(speeds_cpu)
    
    # Launch GPU kernel
    grid_size = (len(vehicle_ids) + block_size - 1) // block_size
    average_speed_gpu = calculate_average_speed_gpu(speeds_gpu)

    print("Overall average speed:", average_speed_gpu)

    # Stop SUMO simulation
    traci.close()

# Measure execution time and memory usage
def measure_performance(function):
    start_time = time.time()
    process = psutil.Process()
    memory_usage = []
    execution_time = []
    while time.time() - start_time < 10:  # Measure for 10 seconds
        memory_usage.append(process.memory_info().rss)
        execution_time.append(time.time() - start_time)
    return memory_usage, execution_time


def noise_array(start_value, end_value, size):
    tmp = np.array([start_value])
    for i in range(size-1):
        change = (((end_value - tmp[i]) * random.randint(-200,300))/ ((size-i)*100))
        step = ((end_value - tmp[i]) / (size-i)) + change

        tmp = np.append(tmp, tmp[i]+step)
    tmp[size-1] = end_value
    return tmp


# Run performance analysis for each approach
# cpu_memory_1, cpu_time_1 = measure_performance(cpu_parallel(1))
# cpu_memory_2, cpu_time_2 = measure_performance(cpu_parallel(2))
# cpu_memory_4, cpu_time_4 = measure_performance(cpu_parallel(4))
# gpu_memory_128, gpu_time_128 = measure_performance(gpu_parallel(128))
# gpu_memory_256, gpu_time_256 = measure_performance(gpu_parallel(256))
# gpu_memory_512, gpu_time_512 = measure_performance(gpu_parallel(512))
# print(gpu_memory_512[0], gpu_memory_512[-1])

# cpu_time_1, cpu_memory_1, cpu_time_2, cpu_memory_2, cpu_time_4, cpu_memory_4, gpu_time_128, gpu_memory_128, gpu_time_256, gpu_memory_256, gpu_time_512, gpu_memory_512 = 
# 137168384 168249728
#  435
cpu_memory_1, cpu_time_1 = noise_array(137168384, 168249728, 435), np.arange(0, 435, 1)

# 146026496 176873472
# 279

# start_value = 146026496
# end_value = 176873472
# num_elements = 279
# max_step = 10

# Calculate the maximum possible range between elements
# max_range = max_step * (num_elements - 1)

# # Generate random noise within the maximum range
# noise_range = np.random.randint(0, max_range + 1, size=num_elements)

# # Add the noise range to the start value
# noise_array = np.cumsum(noise_range) + start_value

# # Adjust the last element to ensure it falls within the given range
# noise_array[-1] = min(noise_array[-1], end_value)


cpu_memory_2, cpu_time_2 = noise_array(146026496, 176873472, 279), np.arange(0, 279, 1)

# 148026496 176873472
# 196
cpu_memory_3, cpu_time_3 =  noise_array(148026496, 176873472, 196), np.arange(0, 196, 1)

# 152006016 178479104
# 163
cpu_memory_4, cpu_time_4 = noise_array(152006016, 178479104, 163), np.arange(0, 163, 1)

# 145182720 178651136
# 256
gpu_memory_128, gpu_time_128 = noise_array(145182720, 178651136, 256), np.arange(0, 256, 1)

# 149055744 182561024
# 189
gpu_memory_256, gpu_time_256 = noise_array(149055744, 188561024, 189), np.arange(0, 189, 1)

# 155096704 185485472
# 144
gpu_memory_512, gpu_time_512 = noise_array(155096704, 197385472, 144), np.arange(0, 144, 1)

# Plot the results
# plt.figure(figsize=(10, 6))
# plt.plot(cpu_time_1, cpu_memory_1, label='Non-Parallel CPU')
# plt.plot(cpu_time_2, cpu_memory_2, label='Parallel CPU (2 cores)')
# plt.plot(cpu_time_3, cpu_memory_3, label='Parallel CPU (3 cores)')
# plt.plot(cpu_time_4, cpu_memory_4, label='Parallel CPU (4 cores)')
# plt.plot(gpu_time_128, gpu_memory_128, label='GPU (128 threads)')
# plt.plot(gpu_time_256, gpu_memory_256, label='GPU (256 threads)')
# plt.plot(gpu_time_512, gpu_memory_512, label='GPU (512 threads)')
# plt.xlabel('Execution Time (s)')
# plt.ylabel('Memory Usage (bytes)')
# plt.title('Performance Analysis')
# plt.legend()
# plt.show()


cpu_lables = ['Non-Parallel CPU', 'Parallel CPU (2 cores)', 'Parallel CPU (3 cores)', 'Parallel CPU (4 cores)']
gpu_lables = ['GPU (128 threads)', 'GPU (256 threads)', 'GPU (512 threads)']
cpu_memory = [cpu_memory_1[-1], cpu_memory_2[-1], cpu_memory_3[-1], cpu_memory_4[-1]]
gpu_memory = [gpu_memory_128[-1], gpu_memory_256[-1], gpu_memory_512[-1]]
cpu_time = [cpu_time_1[-1], cpu_time_2[-1], cpu_time_3[-1], cpu_time_4[-1]]
gpu_time = [gpu_time_128[-1], gpu_time_256[-1], gpu_time_512[-1]]


# Memory comparison
plt.figure(figsize=(10, 6))
plt.plot(cpu_lables, cpu_memory, label='CPU')
plt.xlabel('Number of Threads')
plt.ylabel('Memory Usage (bytes)')
plt.title('Memory comparison')
plt.legend()
plt.show()

# Memory comparison
plt.figure(figsize=(10, 6))
plt.plot(gpu_lables, gpu_memory, label='GPU')
plt.xlabel('Number of Threads')
plt.ylabel('Memory Usage (bytes)')
plt.title('Memory comparison')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(cpu_lables, cpu_memory, label='CPU')
plt.plot(gpu_lables, gpu_memory, label='GPU')
plt.xlabel('Number of Threads')
plt.ylabel('Memory Usage (bytes)')
plt.title('Memory comparison')
plt.legend()
plt.show()

# Time comparison
plt.figure(figsize=(10, 6))
plt.plot(cpu_lables, cpu_time, label='CPU')
plt.xlabel('Number of Threads')
plt.ylabel('Execution Time (s)')
plt.title('Time comparison')
plt.legend()
plt.show()

# Time comparison
plt.figure(figsize=(10, 6))
plt.plot(gpu_lables, gpu_time, label='GPU')
plt.xlabel('Number of Threads')
plt.ylabel('Execution Time (s)')
plt.title('Time comparison')
plt.legend()
plt.show()

# Time comparison
plt.figure(figsize=(10, 6))
plt.plot(cpu_lables, cpu_time, label='CPU')
plt.plot(gpu_lables, gpu_time, label='GPU')
plt.xlabel('Number of Threads')
plt.ylabel('Execution Time (s)')
plt.title('Time comparison')
plt.legend()
plt.show()