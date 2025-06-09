import traci
import json
import re
from collections import OrderedDict

# Start SUMO without RL control
sumoCmd = ["sumo", "-c", "network.sumocfg"]  # Change to "sumo-gui" if you want to see the simulation
traci.start(sumoCmd)

# Dictionary to store actual arrival times
bus_schedule = {}

# Run simulation
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    
    # Get list of all active buses
    buses = [bus for bus in traci.vehicle.getIDList() if traci.vehicle.getTypeID(bus) == "bus"]

    for bus in buses:
        next_stops = traci.vehicle.getNextStops(bus)
        if next_stops:
            stop_id = next_stops[0][2]  # Stop ID
            stop_edge = next_stops[0][0].split("_")[0] # Stop Edge ID
            stop_index = next_stops[0][1]
            departure_time = traci.vehicle.getDeparture(bus)
            current_time = traci.simulation.getTime()
            current_edge = traci.vehicle.getRoadID(bus)
            arrival_time = round(traci.vehicle.getStopArrivalDelay(bus))  # SUMO estimated travel time

            if bus not in bus_schedule:
                bus_schedule[bus] = {"Departure": round(departure_time), "Stop": {}}

            # Log the first time the bus reaches this stop
            if stop_id not in bus_schedule[bus]["Stop"]:
                bus_schedule[bus]["Stop"] = stop_id
                bus_schedule[bus]["Departure"] = round(departure_time)
                bus_schedule[bus]["Arrival delays"] = arrival_time

traci.close()
# Function to extract numerical part of bus ID for sorting
def extract_bus_number(bus_id):
    return int(re.search(r'\d+', bus_id).group())  # Extracts the numeric part

# Sort buses numerically by extracting the number part of the ID
bus_arrival_times = OrderedDict(sorted(bus_schedule.items(), key=lambda x: extract_bus_number(x[0])))
# Save the recorded schedule to a JSON file
with open("bus_timetable.json", "w") as f:
    json.dump(bus_arrival_times, f, indent=4)

print("Bus arrival times have been saved to 'bus_timetable.json'.")
