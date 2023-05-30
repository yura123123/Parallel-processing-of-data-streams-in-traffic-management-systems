import traci
import multiprocessing

def calculate_average_speed(vehicle_ids):
    speeds = [traci.vehicle.getSpeed(v) for v in vehicle_ids]
    return sum(speeds) / len(speeds)

def process_subset(vehicle_subset):
    with traci.connection.RequestThreaded() as traci_conn:
        traci_conn.simulationStep()
        return calculate_average_speed(vehicle_subset)

def parallel_average_speed():
    vehicle_ids = traci.vehicle.getIDList()
    num_processors = multiprocessing.cpu_count()
    subset_size = len(vehicle_ids) // num_processors

    subsets = [vehicle_ids[i:i+subset_size] for i in range(0, len(vehicle_ids), subset_size)]
    
    with multiprocessing.Pool(processes=num_processors) as pool:
        results = pool.map(process_subset, subsets)
    
    overall_avg_speed = sum(results) / len(results)
    print("Overall average speed:", overall_avg_speed)

# Start SUMO simulation
# sumo_binary = "sumo"
# sumo_config = "path/to/your/sumo/config/file.sumocfg"
# sumo_cmd = [sumo_binary, "-c", sumo_config]
# traci.start(sumo_cmd)

# # Run parallel algorithm to find average speed
# parallel_average_speed()

# # Stop SUMO simulation
# traci.close()

num_processors = multiprocessing.cpu_count()
print("Number of processors:", num_processors)