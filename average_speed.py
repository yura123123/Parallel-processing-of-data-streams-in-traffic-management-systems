import traci

# Start the SUMO simulation
sumo_binary = "sumo"  # Path to the SUMO binary
sumo_config = "simple.sumocfg"
sumo_cmd = [sumo_binary, "-c", sumo_config]
traci.start(sumo_cmd)

# Run the simulation for a certain number of steps
simulation_steps = 1000
for step in range(simulation_steps):
    # Advance the simulation by one step
    traci.simulationStep()

    # Access vehicle data
    vehicle_ids = traci.vehicle.getIDList()
    positions = [traci.vehicle.getPosition(v) for v in vehicle_ids]
    speeds = [traci.vehicle.getSpeed(v) for v in vehicle_ids]
    # Perform operations with the vehicle data
    # Example: Calculate average speed
    avg_speed = sum(speeds) / len(speeds)
    # print("Average speed:", avg_speed)

# Stop the SUMO simulation

traci.close()