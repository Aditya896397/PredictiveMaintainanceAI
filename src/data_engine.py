import pandas as pd
import numpy as np
import networkx as nx

class DataEngine:
    def __init__(self):
        self.machines = [
            {"id": "M1", "name": "CNC Router", "type": "Milling", "base_temp": 60, "base_vib": 0.5},
            {"id": "M2", "name": "Conveyor Belt A", "type": "Transport", "base_temp": 45, "base_vib": 0.2},
            {"id": "M3", "name": "Assembly Arm", "type": "Robotics", "base_temp": 50, "base_vib": 0.1},
            {"id": "M4", "name": "Packaging Unit", "type": "Packaging", "base_temp": 40, "base_vib": 0.3},
            {"id": "M5", "name": "Quality Scanner", "type": "Inspection", "base_temp": 35, "base_vib": 0.05}
        ]
        
        # Define Topology M1 -> M2 -> M3 -> M4 -> M5
        self.graph = nx.DiGraph()
        for idx, m in enumerate(self.machines):
            self.graph.add_node(m["id"], **m)
            if idx > 0:
                self.graph.add_edge(self.machines[idx-1]["id"], m["id"])

        self.current_state = {}
        self.reset_state()

    def reset_state(self):
        for m in self.machines:
            self.current_state[m["id"]] = {
                "temperature": m["base_temp"] + np.random.normal(0, 2),
                "vibration": m["base_vib"] + np.random.normal(0, 0.05),
                "pressure": 100 + np.random.normal(0, 5),
                "fault_injected": False
            }
            
    def step(self):
        # Normal fluctuation
        for m_id, state in self.current_state.items():
            if not state.get("fault_injected", False):
                m_info = next(m for m in self.machines if m["id"] == m_id)
                # slowly revert towards base
                state["temperature"] = state["temperature"] * 0.9 + (m_info["base_temp"] + np.random.normal(0, 2)) * 0.1
                state["vibration"] = state["vibration"] * 0.9 + (m_info["base_vib"] + np.random.normal(0, 0.05)) * 0.1
                state["pressure"] = state["pressure"] * 0.9 + (100 + np.random.normal(0, 5)) * 0.1
            else:
                # If fault injected, keep it erratic
                state["temperature"] += np.random.normal(0.5, 2)
                state["vibration"] += np.random.normal(0.05, 0.1)

        # Risk Propagation (simplified) cascasing effect
        edges = list(self.graph.edges)
        for u, v in edges:
            if self.current_state[u].get("fault_injected", False) or self.current_state[u]["temperature"] > 80:
                # propagate some stress to v
                self.current_state[v]["temperature"] += np.random.uniform(0.1, 1.0)
                self.current_state[v]["vibration"] += np.random.uniform(0.01, 0.05)

    def inject_fault(self, machine_id, fault_type, intensity):
        """
        fault_type: 'temperature' or 'vibration'
        intensity: float multiplier
        """
        self.current_state[machine_id][fault_type] *= (1 + intensity)
        self.current_state[machine_id]["fault_injected"] = True

    def clear_fault(self, machine_id):
        self.current_state[machine_id]["fault_injected"] = False

    def get_telemetry_df(self):
        records = []
        for m_id, state in self.current_state.items():
            record = {"machine_id": m_id}
            record.update(state)
            records.append(record)
        return pd.DataFrame(records)
