import os
import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import traci
import matplotlib.pyplot as plt
from sumolib import checkBinary

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialize SUMO (Non-GUI for faster training)
sumoBinary = checkBinary("sumo-gui")
traci.start([
    sumoBinary,
    "-c", "network.sumocfg",
    "--log", "sumo_log.txt",
    "--message-log", "sumo_messages.txt"
])
episode_rewards = []
avg_bus_delays = []

# DQN Parameters
num_actions = 2  # Two phases (NS / EW)
state_size = 9   # Queue Lengths (4) + Bus Delays(4) + Current phase (1)
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
learning_rate = 0.001
batch_size = 32
alpha = 10  # Weight for bus lateness
beta = 2    # Weight for congestion
memory = []

tls_id = "J30"  # Replace with other TLS IDs in your network that you want to test
# "clusterJ48_J49_J50_J51","clusterJ56_J57_J58_J59","clusterJ60_J61_J62_J63","clusterJ68_J69_J70_J71","clusterJ72_J73_J74_J75", "clusterJ34_J35_J36_J37"
def build_dqn():
    model = Sequential([
        Dense(64, input_dim=state_size, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_actions, activation='linear')
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    return model
dqn_model = build_dqn()
target_model = build_dqn()
target_model.set_weights(dqn_model.get_weights())  # Sync weights


def get_buses_at_traffic_light(tls_id):
    buses_at_tl = {}
    controlled_lanes = list(set(traci.trafficlight.getControlledLanes(tls_id)))

    for lane in controlled_lanes:
        buses = [
            veh for veh in traci.lane.getLastStepVehicleIDs(lane)
            if traci.vehicle.getTypeID(veh) == "bus"
        ]
        if buses:
            buses_at_tl[lane] = buses

    return buses_at_tl  # Format: {laneID: [bus1, bus2, ...]}

# def calculate_lane_priority(tls_id):
#     buses_at_tl = get_buses_at_traffic_light(tls_id)
#     controlled_lanes = list(set(traci.trafficlight.getControlledLanes(tls_id)))

#     congestion_data = {lane: traci.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes}
    
#     lane_priority = {}
#     for lane in controlled_lanes:
#         congestion_level = congestion_data[lane]
#         max_bus_delay = 0

#         if lane in buses_at_tl:
#             for bus in buses_at_tl[lane]:
#                 bus_delay = traci.vehicle.getStopArrivalDelay(bus)
#                 max_bus_delay = max(max_bus_delay, bus_delay)

#         lane_priority[lane] = (max_bus_delay * alpha) + (congestion_level * beta)
    
#     return lane_priority

def get_state():
    state = []
    max_bus_delay = 0
    buses_at_tl = get_buses_at_traffic_light(tls_id)
    controlled_lanes = list(set(traci.trafficlight.getControlledLanes(tls_id)))
    congestion_data = {lane: traci.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes}
    for lane in controlled_lanes:
        congestion_level = congestion_data[lane]
        state.append(congestion_level)
    for lane in controlled_lanes:
        if lane in buses_at_tl:
            for bus in buses_at_tl[lane]:
                bus_delay = traci.vehicle.getStopArrivalDelay(bus)
                max_bus_delay = max(max_bus_delay, bus_delay)
            state.append(max_bus_delay)
        else: 
            state.append(0)
    state.append(traci.trafficlight.getPhase(tls_id))  # Include current phase
    # print("state: ",state)
    return np.array(state).reshape(1, -1)

def calculate_reward():
    total_bus_delay = 0
    bus_count = 0
    controlled_lanes = list(set(traci.trafficlight.getControlledLanes(tls_id)))
    for lane in get_buses_at_traffic_light(tls_id).values():
        for bus in lane:
            bus_delay = traci.vehicle.getStopArrivalDelay(bus)
            total_bus_delay += bus_delay 
            bus_count += 1
        
    total_congestion = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes)
    avg_bus_delay = total_bus_delay / max(bus_count, 1)
    avg_bus_delays.append(avg_bus_delay)
    avg_congestion = total_congestion/4
    congestion_threshold = 3 
    bus_lateness_penalty = 3
    congestion_penalty = 1
    print("average congestion: ", avg_congestion," - threshold 3,  ", "average bus delay: ",avg_bus_delay)
    if total_congestion > congestion_threshold:
        reward = -(bus_lateness_penalty * (avg_bus_delay-120) / 60) - (congestion_penalty * (avg_congestion - congestion_threshold))
    else:
        reward = -(bus_lateness_penalty * (avg_bus_delay-120) / 60)  # No penalty if congestion is below threshold
    
    return reward

def apply_phase(tls_id, action, duration):
    """
    Set traffic light phase and keep it active for a given duration.

    :param tls_id: Traffic light ID
    :param action: Chosen phase (0 or 1)
    :param duration: Number of SUMO simulation steps to hold this phase
    """
    traci.trafficlight.setPhase(tls_id, action)
    # print(f"Traffic light {tls_id} set to phase {action} for {duration} seconds")

    for _ in range(duration):  
        traci.simulationStep()

def choose_action(state):
    if np.random.rand() < epsilon:
        return np.random.randint(num_actions)  # Exploration
    return np.argmax(dqn_model.predict(state, verbose=0))  # Exploitation

def store_experience(state, action, reward, next_state, done):
    
    memory.append((state, action, reward, next_state, done))
    if len(memory) > 10000:
        memory.pop(0)

def train_dqn():
    if len(memory) < batch_size:
        return

    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward if done else reward + gamma * np.amax(target_model.predict(next_state, verbose=0))
        target_q_values = dqn_model.predict(state, verbose=0)
        target_q_values[0][action] = target
        dqn_model.fit(state, target_q_values, epochs=1, verbose=0)

try:
    for episode in range(60):
        traci.load(["-c", "network.sumocfg"])  
        state = get_state()
        total_reward = 0

        for step in range(60): 
            done = traci.simulation.getMinExpectedNumber() == 0
            action = choose_action(state)
            
            try:
                apply_phase(tls_id, action, duration=10)
                traci.simulationStep()
                next_state = get_state()
                reward = calculate_reward()
            except Exception as step_err:
                print(f" Error during simulation step at Episode {episode}, Step {step}: {step_err}")
                break

            store_experience(state, action, reward, next_state, done)

            if step % 10 == 0:
                train_dqn()

            state = next_state
            total_reward += reward

            if step % 10 == 0:
                print(f"Episode {episode}, Step {step}: Action={action}, Reward={reward}, Total={total_reward}")

        episode_rewards.append(total_reward)
        if episode % 10 == 0:
            target_model.set_weights(dqn_model.get_weights())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

except Exception as e:
    print(f" Training failed with error: {e}")

finally:
    traci.close()
    print(" TraCI connection closed safely.")

dqn_model.save("trained_dqn_model.h5")

traci.close()

#  Plot Training Results
plt.figure(figsize=(12, 5))

# Plot Total Reward per Episode
plt.subplot(1, 2, 1)
plt.plot(episode_rewards, label="Total Reward",color="yellow")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode")
plt.legend()

# Plot Bus Delay per Episode
plt.subplot(1, 2, 2)
plt.plot(avg_bus_delays, label="Total Bus Delay", color="red")
plt.xlabel("Episode")
plt.ylabel("Total Bus Delay (s)")
plt.title("Bus Delay per Step per Episode")
plt.legend()

plt.tight_layout()
plt.show()