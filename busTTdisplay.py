import json
import pandas as pd
import traci

# Load bus arrival times from JSON file
with open("bus_timetable.json", "r") as f:
    bus_arrival_times = json.load(f)

# Convert JSON to a list of dictionaries
data = []
for bus_id, stops_dict in bus_arrival_times.items():
    for stop_id, arrival_time in stops_dict.items():
        data.append({"Bus ID": bus_id, "Stop ID": stop_id, "Depart Time (s)":traci.vehicle.getDeparture(bus_id), "Arrival Delay (s)": arrival_time})

# Convert to Pandas DataFrame
df = pd.DataFrame(data)

# Sort the table by Bus ID and Stop ID
df.sort_values(by=["Bus ID", "Stop ID"], inplace=True)

# Display the table in VS Code
print(df)