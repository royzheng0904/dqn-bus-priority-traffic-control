import traci
from sumolib import checkBinary
import matplotlib.pyplot as plt
import numpy as np
import time

# Settings
tls_ids = {
    "J30": 0,
    "clusterJ34_J35_J36_J37": 0
}
bus_type = "bus"
check_interval = 1  # second
bus_detection_range = 100  # meters

# SUMO startup
sumoBinary = checkBinary("sumo")
traci.start([sumoBinary, "-c", "network.sumocfg"])

# Metrics per step
avg_delay_per_step = []
step_counter = 0

def bus_approaching(lane_id):
    for veh_id in traci.lane.getLastStepVehicleIDs(lane_id):
        if traci.vehicle.getTypeID(veh_id) == bus_type:
            distance = traci.vehicle.getLanePosition(veh_id)
            lane_length = traci.lane.getLength(lane_id)
            if lane_length - distance <= bus_detection_range:
                return True
    return False

def get_current_avg_bus_delay():
    delays = []
    for veh_id in traci.vehicle.getIDList():
        if traci.vehicle.getTypeID(veh_id) == bus_type:
            try:
                delays.append(traci.vehicle.getStopArrivalDelay(veh_id))
            except traci.exceptions.TraCIException:
                continue
    return np.mean(delays) if delays else 0

def control_all_tls():
    global step_counter
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step_counter += 1

        # Switch phase if bus is near
        for tls_id, green_phase_index in tls_ids.items():
            controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
            if any(bus_approaching(lane) for lane in controlled_lanes):
                if traci.trafficlight.getPhase(tls_id) != green_phase_index:
                    traci.trafficlight.setPhase(tls_id, green_phase_index)

        # Record average bus delay at this step
        avg_delay = get_current_avg_bus_delay()
        avg_delay_per_step.append(avg_delay)

        time.sleep(check_interval)

    traci.close()

# Run control logic
control_all_tls()

# Plotting (like in RL)
plt.figure(figsize=(10, 5))
plt.plot(avg_delay_per_step, label="Avg. Bus Delay", color="red")
plt.xlabel("Timestep")
plt.ylabel("Bus Delay (s)")
plt.title("Bus Delay per Step per Episode")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
