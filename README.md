# DQN-Based Traffic Signal Control for Bus Prioritization

This project implements a Deep Q-Network (DQN) reinforcement learning model to optimize traffic signal timings with a focus on prioritizing delayed public buses, using the SUMO traffic simulator.

## Objective

To investigate whether reinforcement learning can serve as a viable approach for intelligent traffic signal control—particularly one that responds to real-time traffic conditions while prioritizing delayed public buses.

## Reinforcement Learning Details

- **Algorithm**: Deep Q-Network (DQN)
- **Environment**: SUMO (Simulation of Urban Mobility)
- **State Space**:
  - Queue lengths (N, E, S, W)
  - Average Bus lateness in seconds(At the traffic lights)
  - Current traffic phase
- **Action Space**: Discrete phase choices (e.g., NS or EW green)
- **Reward Function**:
  - Penalizes bus lateness (weighted higher)
  - Penalizes traffic congestion (secondary)

## System Architecture

- **Simulation Control**: Python with TraCI API
- **Machine Learning**: TensorFlow/Keras
- **Visualisation**: Matplotlib
- **Evaluation Metrics**:
  - Average Bus Delay
  - Congestion Level
  - Total Episode Reward (RL only)

## Project Structure

```
├── rlAgent.py                 # Multi-agent DQN controller
├── testfile.py                # Single-agent test
├── busTT.py                   # Timetable extractor
├── busTTdisplay.py            # Timetable visualiser
├── bus_priority_no_agent.py   # Rule-based controller
├── bus_timetable.json         # Bus travel time logs
├── busstop.xml                # Bus stop definitions
├── net.net.xml                # Road network
├── network.rou.xml            # Vehicle routes
├── network.sumocfg            # SUMO config
└── README.md
```

## Running the Project

Train multi-agent DQN:
```bash
python rlAgent.py
```

Run a rule-based baseline:
```bash
python bus_priority_no_agent.py
```


## Requirements

Install Python dependencies:
```bash
pip install -r requirements.txt
```


## Results Summary

- The DQN agent shows a steady reduction in congestion and an increase in total reward over episodes.
- Although the rule-based controller achieves lower bus delays, it increases congestion.
- The RL-based method balances efficiency and fairness and shows great promise with more training.

##  Future Work

- Multi-agent coordination for network-wide optimisation
- Integration with real-time GPS data
- Deployment in complex or real-world traffic networks

