import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

# Disk Scheduling Algorithms
def fcfs(requests, head):
    sequence = [head] + requests
    total_distance = sum(abs(sequence[i] - sequence[i-1]) for i in range(1, len(sequence)))
    return sequence, total_distance

def sstf(requests, head):
    sequence, total_distance = [head], 0
    requests = sorted(requests)
    while requests:
        closest = min(requests, key=lambda x: abs(head - x))
        total_distance += abs(head - closest)
        head = closest
        sequence.append(closest)
        requests.remove(closest)
    return sequence, total_distance

def scan(requests, head, disk_size, direction="right"):
    sequence = [head]
    left = [r for r in requests if r < head] + [0]
    right = [r for r in requests if r > head] + [disk_size - 1]
    if direction == "right":
        sequence += sorted(right) + sorted(left, reverse=True)
    else:
        sequence += sorted(left, reverse=True) + sorted(right)
    total_distance = sum(abs(sequence[i] - sequence[i-1]) for i in range(1, len(sequence)))
    return sequence, total_distance

# AI-Based Optimization (DQN)
class DQNScheduler:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.model = self.build_model()
    
    def build_model(self):
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.001))
        return model
    
    def act(self, state):
        return random.choice(range(self.action_size)) if np.random.rand() < 0.1 else np.argmax(self.model.predict(state))

# Visualization
def visualize(sequence):
    plt.figure(figsize=(10, 4))
    plt.plot(sequence, range(len(sequence)), marker='o', linestyle='-', color='b')
    plt.xlabel("Disk Cylinders")
    plt.ylabel("Request Execution Order")
    plt.title("Disk Scheduling Simulation")
    plt.grid()
    st.pyplot(plt)

# Streamlit UI
st.title("AI-Based Disk Scheduling Simulator")
requests = st.text_input("Enter Disk Requests (comma-separated):")
head = st.number_input("Initial Head Position", min_value=0, max_value=200, value=53)
algo = st.selectbox("Choose Algorithm", ["FCFS", "SSTF", "SCAN"])

disk_size = 200  # Disk size assumed

def run_simulation():
    if requests:
        req_list = list(map(int, requests.split(',')))
        if algo == "FCFS":
            sequence, total_distance = fcfs(req_list, head)
        elif algo == "SSTF":
            sequence, total_distance = sstf(req_list, head)
        else:
            sequence, total_distance = scan(req_list, head, disk_size)
        st.write(f"Total Seek Distance: {total_distance}")
        visualize(sequence)

if st.button("Run Simulation"):
    run_simulation()
