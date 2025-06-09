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

# Traffic light settings
tls_ids = ["J30","clusterJ48_J49_J50_J51","clusterJ56_J57_J58_J59","clusterJ60_J61_J62_J63","clusterJ68_J69_J70_J71","clusterJ72_J73_J74_J75", "clusterJ34_J35_J36_J37"]
num_actions = 2
state_size = 9
gamma = 0.9
epsilon_decay = 0.995
epsilon_min = 0.01
learning_rate = 0.01
batch_size = 32
num_episodes = 50
max_steps = 70

# Store each agent's models, memory, epsilon
agents = {
    tls_id: {
        "model": None,
        "target_model": None,
        "memory": [],
        "epsilon": 1.0,
        "episode_rewards": [],
        "episode_delays": []
    } for tls_id in tls_ids
}

def build_dqn():
    model = Sequential([
        Dense(64, input_dim=state_size, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_actions, activation='linear')
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    return model

for tls_id in tls_ids:
    agents[tls_id]["model"] = build_dqn()
    agents[tls_id]["target_model"] = build_dqn()
    agents[tls_id]["target_model"].set_weights(agents[tls_id]["model"].get_weights())

# SUMO setup
sumoBinary = checkBinary("sumo")

def get_buses_at_traffic_light(tls_id):
    buses_at_tl = {}
    controlled_lanes = list(set(traci.trafficlight.getControlledLanes(tls_id)))
    for lane in controlled_lanes:
        buses = [veh for veh in traci.lane.getLastStepVehicleIDs(lane) if traci.vehicle.getTypeID(veh) == "bus"]
        if buses:
            buses_at_tl[lane] = buses
    return buses_at_tl

def get_state(tls_id):
    state = []
    max_bus_delay = 0
    buses_at_tl = get_buses_at_traffic_light(tls_id)
    controlled_lanes = list(set(traci.trafficlight.getControlledLanes(tls_id)))
    congestion_data = {lane: traci.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes}
    for lane in controlled_lanes:
        state.append(congestion_data[lane])
    for lane in controlled_lanes:
        if lane in buses_at_tl:
            for bus in buses_at_tl[lane]:
                bus_delay = traci.vehicle.getStopArrivalDelay(bus)
                max_bus_delay = max(max_bus_delay, bus_delay)
            state.append(max_bus_delay)
        else:
            state.append(0)
    state.append(traci.trafficlight.getPhase(tls_id))
    return np.array(state).reshape(1, -1)

def calculate_reward(tls_id):
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
    avg_congestion = total_congestion / 4
    congestion_threshold = 3 
    bus_lateness_penalty = 3
    congestion_penalty = 1
    if total_congestion > congestion_threshold:
        reward = -(bus_lateness_penalty * (avg_bus_delay - 120) / 60) - (congestion_penalty * (avg_congestion - congestion_threshold))
    else:
        reward = -(bus_lateness_penalty * (avg_bus_delay - 120) / 60)
    return reward, avg_bus_delay

def apply_phase(tls_id, action, duration):
    traci.trafficlight.setPhase(tls_id, action)
    for _ in range(duration):  
        traci.simulationStep()

def store_experience(agent, state, action, reward, next_state):
    agent["memory"].append((state, action, reward, next_state, False))
    if len(agent["memory"]) > 10000:
        agent["memory"].pop(0)

# Metrics tracking
avg_episode_rewards = []
avg_episode_bus_delays = []

# Start SUMO and begin training
traci.start([sumoBinary, "-c", "network.sumocfg"])

for episode in range(num_episodes):
    traci.load(["-c", "network.sumocfg"])
    states = {tls_id: get_state(tls_id) for tls_id in tls_ids}
    episode_reward_sum = 0
    episode_delay_sum = 0
    for step in range(max_steps):
        for tls_id in tls_ids:
            agent = agents[tls_id]
            state = states[tls_id]
            if np.random.rand() < agent["epsilon"]:
                action = np.random.randint(num_actions)
            else:
                action = np.argmax(agent["model"].predict(state, verbose=0))
            apply_phase(tls_id, action, duration=10)
            next_state = get_state(tls_id)
            reward, avg_delay = calculate_reward(tls_id)
            print(f"[Ep {episode} | Step {step}] {tls_id} → Action: {action}, Reward: {reward:.2f}, AvgDelay: {avg_delay:.2f}")

            agent["memory"].append((state, action, reward, next_state, False))
            if len(agent["memory"]) > 10000:
                agent["memory"].pop(0)

            if len(agent["memory"]) >= batch_size:
                minibatch = random.sample(agent["memory"], batch_size)
                for s, a, r, s_, done in minibatch:
                    target = r + gamma * np.amax(agent["target_model"].predict(s_, verbose=0))
                    target_q = agent["model"].predict(s, verbose=0)
                    target_q[0][a] = target
                    agent["model"].fit(s, target_q, epochs=1, verbose=0)

            states[tls_id] = next_state
            episode_reward_sum += reward
            episode_delay_sum += avg_delay

        # Sync target model
        if step % 10 == 0:
            for tls_id in tls_ids:
                agents[tls_id]["target_model"].set_weights(agents[tls_id]["model"].get_weights())

    # Decay epsilon
    for tls_id in tls_ids:
        agents[tls_id]["epsilon"] = max(epsilon_min, agents[tls_id]["epsilon"] * epsilon_decay)

    # Track average metrics
    avg_episode_rewards.append(episode_reward_sum)
    avg_episode_bus_delays.append(episode_delay_sum)

# Save trained models
for tls_id in tls_ids:
    agents[tls_id]["model"].save(f"trained_dqn_{tls_id}.h5")

traci.close()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(avg_episode_rewards, label="Avg. Total Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Average Reward per Episode")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(avg_episode_bus_delays, label="Avg. Bus Delay (s)", color="red")
plt.xlabel("Timestep")
plt.ylabel("Delay (s)")
plt.title("Bus Delay per Step per episode")
plt.legend()

plt.tight_layout()
plt.show()
