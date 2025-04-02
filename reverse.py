# import streamlit as st
# import matplotlib.pyplot as plt
# import numpy as np
# import random
# from collections import deque
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# import tensorflow as tf

# # Disk Scheduling Algorithms
# def fcfs(requests, head):
#     sequence = [head] + requests
#     total_distance = sum(abs(sequence[i] - sequence[i-1]) for i in range(1, len(sequence)))
#     return sequence, total_distance

# def sstf(requests, head):
#     sequence, total_distance = [head], 0
#     requests = sorted(requests)
#     while requests:
#         closest = min(requests, key=lambda x: abs(head - x))
#         total_distance += abs(head - closest)
#         head = closest
#         sequence.append(closest)
#         requests.remove(closest)
#     return sequence, total_distance

# def scan(requests, head, disk_size, direction="right"):
#     sequence = [head]
#     left = [r for r in requests if r < head] + [0]
#     right = [r for r in requests if r > head] + [disk_size - 1]
#     if direction == "right":
#         sequence += sorted(right) + sorted(left, reverse=True)
#     else:
#         sequence += sorted(left, reverse=True) + sorted(right)
#     total_distance = sum(abs(sequence[i] - sequence[i-1]) for i in range(1, len(sequence)))
#     return sequence, total_distance

# # AI-Based Optimization (DQN)
# class DQNScheduler:
#     def __init__(self, state_size, action_size):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.memory = deque(maxlen=2000)
#         self.model = self.build_model()
    
#     def build_model(self):
#         model = Sequential([
#             Dense(24, input_dim=self.state_size, activation='relu'),
#             Dense(24, activation='relu'),
#             Dense(self.action_size, activation='linear')
#         ])
#         model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.001))
#         return model
    
#     def act(self, state):
#         return random.choice(range(self.action_size)) if np.random.rand() < 0.1 else np.argmax(self.model.predict(state))

# # Visualization
# def visualize(sequence):
#     plt.figure(figsize=(10, 4))
#     plt.plot(sequence, range(len(sequence)), marker='o', linestyle='-', color='b')
#     plt.xlabel("Disk Cylinders")
#     plt.ylabel("Request Execution Order")
#     plt.title("Disk Scheduling Simulation")
#     plt.grid()
#     st.pyplot(plt)

# # Streamlit UI
# st.title("AI-Based Disk Scheduling Simulator")
# requests = st.text_input("Enter Disk Requests (comma-separated):")
# head = st.number_input("Initial Head Position", min_value=0, max_value=200, value=53)
# algo = st.selectbox("Choose Algorithm", ["FCFS", "SSTF", "SCAN"])

# disk_size = 200  # Disk size assumed

# def run_simulation():
#     if requests:
#         req_list = list(map(int, requests.split(',')))
#         if algo == "FCFS":
#             sequence, total_distance = fcfs(req_list, head)
#         elif algo == "SSTF":
#             sequence, total_distance = sstf(req_list, head)
#         else:
#             sequence, total_distance = scan(req_list, head, disk_size)
#         st.write(f"Total Seek Distance: {total_distance}")
#         visualize(sequence)

# if st.button("Run Simulation"):
#     run_simulation()


# import streamlit as st
# import matplotlib.pyplot as plt
# import numpy as np
# import random
# from collections import deque
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# import tensorflow as tf

# # Disk Scheduling Algorithms
# def fcfs(requests, head):
#     sequence = [head] + requests
#     total_distance = sum(abs(sequence[i] - sequence[i-1]) for i in range(1, len(sequence)))
#     return sequence, total_distance

# def sstf(requests, head):
#     sequence, total_distance = [head], 0
#     requests = sorted(requests)
#     while requests:
#         closest = min(requests, key=lambda x: abs(head - x))
#         total_distance += abs(head - closest)
#         head = closest
#         sequence.append(closest)
#         requests.remove(closest)
#     return sequence, total_distance

# def scan(requests, head, disk_size, direction="right"):
#     sequence = [head]
#     left = [r for r in requests if r < head] + [0]
#     right = [r for r in requests if r > head] + [disk_size - 1]
#     if direction == "right":
#         sequence += sorted(right) + sorted(left, reverse=True)
#     else:
#         sequence += sorted(left, reverse=True) + sorted(right)
#     total_distance = sum(abs(sequence[i] - sequence[i-1]) for i in range(1, len(sequence)))
#     return sequence, total_distance

# def c_scan(requests, head, disk_size):
#     sequence = [head]
#     right = [r for r in requests if r >= head] + [disk_size - 1]
#     left = [r for r in requests if r < head] + [0]
#     sequence += sorted(right) + sorted(left)
#     total_distance = sum(abs(sequence[i] - sequence[i-1]) for i in range(1, len(sequence)))
#     return sequence, total_distance

# def c_look(requests, head):
#     sequence = [head]
#     right = sorted([r for r in requests if r >= head])
#     left = sorted([r for r in requests if r < head])
#     sequence += right + left
#     total_distance = sum(abs(sequence[i] - sequence[i-1]) for i in range(1, len(sequence)))
#     return sequence, total_distance

# # AI-Based Optimization (DQN)
# class DQNScheduler:
#     def __init__(self, state_size, action_size):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.memory = deque(maxlen=2000)
#         self.model = self.build_model()
    
#     def build_model(self):
#         model = Sequential([
#             Dense(24, input_dim=self.state_size, activation='relu'),
#             Dense(24, activation='relu'),
#             Dense(self.action_size, activation='linear')
#         ])
#         model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.001))
#         return model
    
#     def act(self, state):
#         return random.choice(range(self.action_size)) if np.random.rand() < 0.1 else np.argmax(self.model.predict(state))

# # Visualization
# def visualize(sequence):
#     plt.figure(figsize=(10, 4))
#     plt.plot(sequence, range(len(sequence)), marker='o', linestyle='-', color='b')
#     plt.xlabel("Disk Cylinders")
#     plt.ylabel("Request Execution Order")
#     plt.title("Disk Scheduling Simulation")
#     plt.grid()
#     st.pyplot(plt)

# # Streamlit UI
# st.title("AI-Based Disk Scheduling Simulator")

# requests = st.text_input("Enter Disk Requests (comma-separated):")
# head = st.slider("Initial Head Position", min_value=0, max_value=200, value=53)
# algo = st.selectbox("Choose Algorithm", ["FCFS", "SSTF", "SCAN", "C-SCAN", "C-LOOK"])

# disk_size = 200  # Disk size assumed

# def run_simulation():
#     if requests:
#         req_list = list(map(int, requests.split(',')))
#         if algo == "FCFS":
#             sequence, total_distance = fcfs(req_list, head)
#         elif algo == "SSTF":
#             sequence, total_distance = sstf(req_list, head)
#         elif algo == "SCAN":
#             sequence, total_distance = scan(req_list, head, disk_size)
#         elif algo == "C-SCAN":
#             sequence, total_distance = c_scan(req_list, head, disk_size)
#         else:
#             sequence, total_distance = c_look(req_list, head)
#         st.write(f"Total Seek Distance: {total_distance}")
#         visualize(sequence)

# if st.button("Run Simulation"):
#     run_simulation()


# import streamlit as st
# import matplotlib.pyplot as plt
# import numpy as np
# import random
# from collections import deque
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# import tensorflow as tf

# # Disk Scheduling Algorithms
# def fcfs(requests, head):
#     sequence = [head] + requests
#     total_distance = sum(abs(sequence[i] - sequence[i-1]) for i in range(1, len(sequence)))
#     return sequence, total_distance

# def sstf(requests, head):
#     sequence, total_distance = [head], 0
#     requests = sorted(requests)
#     while requests:
#         closest = min(requests, key=lambda x: abs(head - x))
#         total_distance += abs(head - closest)
#         head = closest
#         sequence.append(closest)
#         requests.remove(closest)
#     return sequence, total_distance

# def scan(requests, head, disk_size, direction="right"):
#     sequence = [head]
#     left = [r for r in requests if r < head] + [0]
#     right = [r for r in requests if r > head] + [disk_size - 1]
#     if direction == "right":
#         sequence += sorted(right) + sorted(left, reverse=True)
#     else:
#         sequence += sorted(left, reverse=True) + sorted(right)
#     total_distance = sum(abs(sequence[i] - sequence[i-1]) for i in range(1, len(sequence)))
#     return sequence, total_distance

# def c_scan(requests, head, disk_size):
#     sequence = [head]
#     right = [r for r in requests if r >= head] + [disk_size - 1]
#     left = [r for r in requests if r < head] + [0]
#     sequence += sorted(right) + sorted(left)
#     total_distance = sum(abs(sequence[i] - sequence[i-1]) for i in range(1, len(sequence)))
#     return sequence, total_distance

# def c_look(requests, head):
#     sequence = [head]
#     right = sorted([r for r in requests if r >= head])
#     left = sorted([r for r in requests if r < head])
#     sequence += right + left
#     total_distance = sum(abs(sequence[i] - sequence[i-1]) for i in range(1, len(sequence)))
#     return sequence, total_distance

# # Visualization
# def visualize(sequences, algorithms):
#     plt.figure(figsize=(10, 6))
#     colors = ['b', 'g', 'r', 'c', 'm', 'y']
#     for i, (sequence, algo) in enumerate(zip(sequences, algorithms)):
#         plt.plot(sequence, range(len(sequence)), marker='o', linestyle='-', color=colors[i % len(colors)], label=algo)
#     plt.xlabel("Disk Cylinders")
#     plt.ylabel("Request Execution Order")
#     plt.title("Disk Scheduling Simulation")
#     plt.legend()
#     plt.grid()
#     st.pyplot(plt)

# # Streamlit UI
# st.title("AI-Based Disk Scheduling Simulator")

# requests = st.text_input("Enter Disk Requests (comma-separated):")
# head = st.slider("Initial Head Position", min_value=0, max_value=200, value=53)
# selected_algorithms = st.multiselect("Choose Algorithms", ["FCFS", "SSTF", "SCAN", "C-SCAN", "C-LOOK"], default=["FCFS", "SSTF"])

# disk_size = 200  # Disk size assumed

# def run_simulation():
#     if requests:
#         req_list = list(map(int, requests.split(',')))
#         sequences, algorithms = [], []
#         for algo in selected_algorithms:
#             if algo == "FCFS":
#                 sequence, total_distance = fcfs(req_list, head)
#             elif algo == "SSTF":
#                 sequence, total_distance = sstf(req_list, head)
#             elif algo == "SCAN":
#                 sequence, total_distance = scan(req_list, head, disk_size)
#             elif algo == "C-SCAN":
#                 sequence, total_distance = c_scan(req_list, head, disk_size)
#             else:
#                 sequence, total_distance = c_look(req_list, head)
#             st.write(f"{algo} Total Seek Distance: {total_distance}")
#             sequences.append(sequence)
#             algorithms.append(algo)
#         visualize(sequences, algorithms)

# if st.button("Run Simulation"):
#     run_simulation()


# import streamlit as st
# import matplotlib.pyplot as plt
# import numpy as np
# import random
# from collections import deque
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# import tensorflow as tf
# import plotly.express as px
# import plotly.graph_objects as go

# # Disk Scheduling Algorithms
# def fcfs(requests, head):
#     sequence = [head] + requests
#     total_distance = sum(abs(sequence[i] - sequence[i-1]) for i in range(1, len(sequence)))
#     return sequence, total_distance

# def sstf(requests, head):
#     sequence, total_distance = [head], 0
#     requests = sorted(requests)
#     while requests:
#         closest = min(requests, key=lambda x: abs(head - x))
#         total_distance += abs(head - closest)
#         head = closest
#         sequence.append(closest)
#         requests.remove(closest)
#     return sequence, total_distance

# def scan(requests, head, disk_size, direction="right"):
#     sequence = [head]
#     left = [r for r in requests if r < head] + [0]
#     right = [r for r in requests if r > head] + [disk_size - 1]
#     if direction == "right":
#         sequence += sorted(right) + sorted(left, reverse=True)
#     else:
#         sequence += sorted(left, reverse=True) + sorted(right)
#     total_distance = sum(abs(sequence[i] - sequence[i-1]) for i in range(1, len(sequence)))
#     return sequence, total_distance

# def c_scan(requests, head, disk_size):
#     sequence = [head]
#     right = [r for r in requests if r >= head] + [disk_size - 1]
#     left = [r for r in requests if r < head] + [0]
#     sequence += sorted(right) + sorted(left)
#     total_distance = sum(abs(sequence[i] - sequence[i-1]) for i in range(1, len(sequence)))
#     return sequence, total_distance

# def c_look(requests, head):
#     sequence = [head]
#     right = sorted([r for r in requests if r >= head])
#     left = sorted([r for r in requests if r < head])
#     sequence += right + left
#     total_distance = sum(abs(sequence[i] - sequence[i-1]) for i in range(1, len(sequence)))
#     return sequence, total_distance

# # Visualization using Plotly
# def visualize_plotly(sequences, algorithms):
#     fig = go.Figure()
#     colors = ['blue', 'green', 'red', 'purple', 'orange', 'pink']
#     for i, (sequence, algo) in enumerate(zip(sequences, algorithms)):
#         fig.add_trace(go.Scatter(x=sequence, y=list(range(len(sequence))), mode='lines+markers', name=algo, line=dict(color=colors[i % len(colors)], width=2)))
#     fig.update_layout(title='Disk Scheduling Simulation', xaxis_title='Disk Cylinders', yaxis_title='Request Execution Order', template='plotly_dark')
#     st.plotly_chart(fig)

# # Streamlit UI
# st.set_page_config(page_title="AI-Based Disk Scheduling Simulator", layout="wide")
# st.title("🖥️ AI-Based Disk Scheduling Simulator")
# st.sidebar.header("📌 Simulation Settings")

# requests = st.sidebar.text_input("Enter Disk Requests (comma-separated):")
# head = st.sidebar.slider("Initial Head Position", min_value=0, max_value=200, value=53)
# selected_algorithms = st.sidebar.multiselect("Choose Algorithms", ["FCFS", "SSTF", "SCAN", "C-SCAN", "C-LOOK"], default=["FCFS", "SSTF"])

# disk_size = 200  # Disk size assumed

# def run_simulation():
#     if requests:
#         req_list = list(map(int, requests.split(',')))
#         sequences, algorithms, distances = [], [], []
#         for algo in selected_algorithms:
#             if algo == "FCFS":
#                 sequence, total_distance = fcfs(req_list, head)
#             elif algo == "SSTF":
#                 sequence, total_distance = sstf(req_list, head)
#             elif algo == "SCAN":
#                 sequence, total_distance = scan(req_list, head, disk_size)
#             elif algo == "C-SCAN":
#                 sequence, total_distance = c_scan(req_list, head, disk_size)
#             else:
#                 sequence, total_distance = c_look(req_list, head)
#             st.sidebar.write(f"{algo} Total Seek Distance: {total_distance}")
#             sequences.append(sequence)
#             algorithms.append(algo)
#             distances.append(total_distance)
        
#         # Visualization
#         visualize_plotly(sequences, algorithms)
        
#         # Additional Live Graphs
#         st.subheader("📊 Algorithm Comparison")
#         fig_bar = px.bar(x=algorithms, y=distances, labels={'x': "Algorithm", 'y': "Total Seek Distance"}, title="Total Seek Distance per Algorithm", color=algorithms, template='plotly_dark')
#         st.plotly_chart(fig_bar)
        
#         fig_pie = px.pie(names=algorithms, values=distances, title="Seek Distance Distribution", color=algorithms, template='plotly_dark')
#         st.plotly_chart(fig_pie)

# if st.sidebar.button("Run Simulation"):
#     run_simulation()



# import streamlit as st
# import matplotlib.pyplot as plt
# import numpy as np
# import random
# from collections import deque
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# import tensorflow as tf
# import plotly.express as px
# import plotly.graph_objects as go

# # Disk Scheduling Algorithms
# def fcfs(requests, head):
#     sequence = [head] + requests
#     total_distance = sum(abs(sequence[i] - sequence[i-1]) for i in range(1, len(sequence)))
#     return sequence, total_distance

# def sstf(requests, head):
#     sequence, total_distance = [head], 0
#     requests = sorted(requests)
#     while requests:
#         closest = min(requests, key=lambda x: abs(head - x))
#         total_distance += abs(head - closest)
#         head = closest
#         sequence.append(closest)
#         requests.remove(closest)
#     return sequence, total_distance

# def scan(requests, head, disk_size, direction="right"):
#     sequence = [head]
#     left = [r for r in requests if r < head] + [0]
#     right = [r for r in requests if r > head] + [disk_size - 1]
#     if direction == "right":
#         sequence += sorted(right) + sorted(left, reverse=True)
#     else:
#         sequence += sorted(left, reverse=True) + sorted(right)
#     total_distance = sum(abs(sequence[i] - sequence[i-1]) for i in range(1, len(sequence)))
#     return sequence, total_distance

# def c_scan(requests, head, disk_size):
#     sequence = [head]
#     right = [r for r in requests if r >= head] + [disk_size - 1]
#     left = [r for r in requests if r < head] + [0]
#     sequence += sorted(right) + sorted(left)
#     total_distance = sum(abs(sequence[i] - sequence[i-1]) for i in range(1, len(sequence)))
#     return sequence, total_distance

# def c_look(requests, head):
#     sequence = [head]
#     right = sorted([r for r in requests if r >= head])
#     left = sorted([r for r in requests if r < head])
#     sequence += right + left
#     total_distance = sum(abs(sequence[i] - sequence[i-1]) for i in range(1, len(sequence)))
#     return sequence, total_distance

# # Visualization using Plotly
# def visualize_plotly(sequences, algorithms):
#     fig = go.Figure()
#     colors = ['blue', 'green', 'red', 'purple', 'orange', 'pink']
#     for i, (sequence, algo) in enumerate(zip(sequences, algorithms)):
#         fig.add_trace(go.Scatter(x=sequence, y=list(range(len(sequence))), mode='lines+markers', name=algo, line=dict(color=colors[i % len(colors)], width=2)))
#     fig.update_layout(title='Disk Scheduling Simulation', xaxis_title='Disk Cylinders', yaxis_title='Request Execution Order', template='plotly_dark')
#     st.plotly_chart(fig)

# # Streamlit UI
# st.set_page_config(page_title="AI-Based Disk Scheduling Simulator", layout="wide")
# st.title("🖥️ AI-Based Disk Scheduling Simulator")
# st.sidebar.header("📌 Simulation Settings")

# requests = st.sidebar.text_input("Enter Disk Requests (comma-separated):")
# head = st.sidebar.slider("Initial Head Position", min_value=0, max_value=200, value=53)
# selected_algorithms = st.sidebar.multiselect("Choose Algorithms", ["FCFS", "SSTF", "SCAN", "C-SCAN", "C-LOOK"], default=["FCFS", "SSTF"])

# disk_size = 200  # Disk size assumed

# def run_simulation():
#     if requests:
#         req_list = list(map(int, requests.split(',')))
#         sequences, algorithms, distances = [], [], []
#         for algo in selected_algorithms:
#             if algo == "FCFS":
#                 sequence, total_distance = fcfs(req_list, head)
#             elif algo == "SSTF":
#                 sequence, total_distance = sstf(req_list, head)
#             elif algo == "SCAN":
#                 sequence, total_distance = scan(req_list, head, disk_size)
#             elif algo == "C-SCAN":
#                 sequence, total_distance = c_scan(req_list, head, disk_size)
#             else:
#                 sequence, total_distance = c_look(req_list, head)
#             st.sidebar.write(f"{algo} Total Seek Distance: {total_distance}")
#             sequences.append(sequence)
#             algorithms.append(algo)
#             distances.append(total_distance)
        
#         # Visualization
#         visualize_plotly(sequences, algorithms)
        
#         # Additional Live Graphs
#         st.subheader("📊 Algorithm Comparison")
#         fig_bar = px.bar(x=algorithms, y=distances, labels={'x': "Algorithm", 'y': "Total Seek Distance"}, title="Total Seek Distance per Algorithm", color=algorithms, template='plotly_dark')
#         st.plotly_chart(fig_bar)
        
#         fig_pie = px.pie(names=algorithms, values=distances, title="Seek Distance Distribution", color=algorithms, template='plotly_dark')
#         st.plotly_chart(fig_pie)

# if st.sidebar.button("Run Simulation"):
#     run_simulation()


# import streamlit as st
# import matplotlib.pyplot as plt
# import numpy as np
# import random
# from collections import deque
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# import tensorflow as tf
# import plotly.express as px
# import plotly.graph_objects as go

# # Disk Scheduling Algorithms
# def fcfs(requests, head):
#     sequence = [head] + requests
#     total_distance = sum(abs(sequence[i] - sequence[i-1]) for i in range(1, len(sequence)))
#     return sequence, total_distance

# def sstf(requests, head):
#     sequence, total_distance = [head], 0
#     requests = sorted(requests)
#     while requests:
#         closest = min(requests, key=lambda x: abs(head - x))
#         total_distance += abs(head - closest)
#         head = closest
#         sequence.append(closest)
#         requests.remove(closest)
#     return sequence, total_distance

# def scan(requests, head, disk_size, direction="right"):
#     sequence = [head]
#     left = [r for r in requests if r < head] + [0]
#     right = [r for r in requests if r > head] + [disk_size - 1]
#     if direction == "right":
#         sequence += sorted(right) + sorted(left, reverse=True)
#     else:
#         sequence += sorted(left, reverse=True) + sorted(right)
#     total_distance = sum(abs(sequence[i] - sequence[i-1]) for i in range(1, len(sequence)))
#     return sequence, total_distance

# def c_scan(requests, head, disk_size):
#     sequence = [head]
#     right = [r for r in requests if r >= head] + [disk_size - 1]
#     left = [r for r in requests if r < head] + [0]
#     sequence += sorted(right) + sorted(left)
#     total_distance = sum(abs(sequence[i] - sequence[i-1]) for i in range(1, len(sequence)))
#     return sequence, total_distance

# def c_look(requests, head):
#     sequence = [head]
#     right = sorted([r for r in requests if r >= head])
#     left = sorted([r for r in requests if r < head])
#     sequence += right + left
#     total_distance = sum(abs(sequence[i] - sequence[i-1]) for i in range(1, len(sequence)))
#     return sequence, total_distance

# # Visualization using Plotly
# def visualize_bar_chart(sequences, algorithms, distances):
#     fig_bar = px.bar(x=algorithms, y=distances, labels={'x': "Algorithm", 'y': "Total Seek Distance"}, title="Total Seek Distance per Algorithm", color=algorithms, template='plotly_dark')
#     st.plotly_chart(fig_bar)

# # Streamlit UI
# st.set_page_config(page_title="AI-Based Disk Scheduling Simulator", layout="wide")
# st.title("🖥️ AI-Based Disk Scheduling Simulator")
# st.sidebar.header("📌 Simulation Settings")

# requests = st.sidebar.text_input("Enter Disk Requests (comma-separated):")
# head = st.sidebar.slider("Initial Head Position", min_value=0, max_value=200, value=53)
# selected_algorithms = st.sidebar.multiselect("Choose Algorithms", ["FCFS", "SSTF", "SCAN", "C-SCAN", "C-LOOK"], default=["FCFS", "SSTF"])

# disk_size = 200  # Disk size assumed

# def run_simulation():
#     if requests:
#         req_list = list(map(int, requests.split(',')))
#         sequences, algorithms, distances = [], [], []
#         for algo in selected_algorithms:
#             if algo == "FCFS":
#                 sequence, total_distance = fcfs(req_list, head)
#             elif algo == "SSTF":
#                 sequence, total_distance = sstf(req_list, head)
#             elif algo == "SCAN":
#                 sequence, total_distance = scan(req_list, head, disk_size)
#             elif algo == "C-SCAN":
#                 sequence, total_distance = c_scan(req_list, head, disk_size)
#             else:
#                 sequence, total_distance = c_look(req_list, head)
#             st.sidebar.write(f"{algo} Total Seek Distance: {total_distance}")
#             sequences.append(sequence)
#             algorithms.append(algo)
#             distances.append(total_distance)
        
#         # Visualization
#         st.subheader("📊 Algorithm Comparison")
#         visualize_bar_chart(sequences, algorithms, distances)

# if st.sidebar.button("Run Simulation"):
#     run_simulation()



import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go

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

def c_scan(requests, head, disk_size):
    sequence = [head]
    right = [r for r in requests if r >= head] + [disk_size - 1]
    left = [r for r in requests if r < head] + [0]
    sequence += sorted(right) + sorted(left)
    total_distance = sum(abs(sequence[i] - sequence[i-1]) for i in range(1, len(sequence)))
    return sequence, total_distance

def c_look(requests, head):
    sequence = [head]
    right = sorted([r for r in requests if r >= head])
    left = sorted([r for r in requests if r < head])
    sequence += right + left
    total_distance = sum(abs(sequence[i] - sequence[i-1]) for i in range(1, len(sequence)))
    return sequence, total_distance

# Visualization using Plotly
def visualize_bar_chart(algorithms, distances):
    fig_bar = px.bar(x=algorithms, y=distances, labels={'x': "Algorithm", 'y': "Total Seek Distance"}, title="Total Seek Distance per Algorithm", color=algorithms, template='plotly_dark')
    st.plotly_chart(fig_bar)

def visualize_pie_chart(algorithms, distances):
    fig_pie = px.pie(values=distances, names=algorithms, title="Proportion of Seek Distances", color=algorithms, template='plotly_dark')
    st.plotly_chart(fig_pie)

# Streamlit UI
st.set_page_config(page_title="AI-Based Disk Scheduling Simulator", layout="wide")
st.title("🖥️ AI-Based Disk Scheduling Simulator")
st.sidebar.header("📌 Simulation Settings")

requests = st.sidebar.text_input("Enter Disk Requests (comma-separated):")
head = st.sidebar.slider("Initial Head Position", min_value=0, max_value=200, value=53)
selected_algorithms = st.sidebar.multiselect("Choose Algorithms", ["FCFS", "SSTF", "SCAN", "C-SCAN", "C-LOOK"], default=["FCFS", "SSTF"])

disk_size = 200  # Disk size assumed

def run_simulation():
    if requests:
        req_list = list(map(int, requests.split(',')))
        sequences, algorithms, distances = [], [], []
        for algo in selected_algorithms:
            if algo == "FCFS":
                sequence, total_distance = fcfs(req_list, head)
            elif algo == "SSTF":
                sequence, total_distance = sstf(req_list, head)
            elif algo == "SCAN":
                sequence, total_distance = scan(req_list, head, disk_size)
            elif algo == "C-SCAN":
                sequence, total_distance = c_scan(req_list, head, disk_size)
            else:
                sequence, total_distance = c_look(req_list, head)
            st.sidebar.write(f"{algo} Total Seek Distance: {total_distance}")
            sequences.append(sequence)
            algorithms.append(algo)
            distances.append(total_distance)
        
        # Visualization
        st.subheader("📊 Algorithm Comparison")
        visualize_bar_chart(algorithms, distances)
        visualize_pie_chart(algorithms, distances)

if st.sidebar.button("Run Simulation"):
    run_simulation()
