import streamlit as st
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

from src.data_engine import DataEngine
from src.ml_core import MLDetector

# Page Config
st.set_page_config(page_title="AgentOps | AI Dashboard", layout="wide", initial_sidebar_state="expanded")

# Load CSS
with open('assets/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize Session State
if 'data_engine' not in st.session_state:
    st.session_state.data_engine = DataEngine()
if 'ml_detector' not in st.session_state:
    st.session_state.ml_detector = MLDetector()
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

engine = st.session_state.data_engine
detector = st.session_state.ml_detector

# Sidebar: Digital Twin Sandbox
with st.sidebar:
    st.markdown("<div class='sidebar-title'>Digital Twin Control</div>", unsafe_allow_html=True)
    st.markdown("Inject simulated faults into the production line to see how AgentOps responds.")
    
    selected_machine_id = st.selectbox("Select Target Machine", [m['id'] for m in engine.machines])
    fault_type = st.radio("Fault Type", ["temperature", "vibration"])
    intensity = st.slider("Fault Intensity Override", 0.0, 3.0, 0.0, 0.1)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Inject Fault", use_container_width=True):
            engine.inject_fault(selected_machine_id, fault_type, intensity)
            st.success("Fault injected.")
    with col2:
        if st.button("Clear Fault", use_container_width=True):
            engine.clear_fault(selected_machine_id)
            st.success("Fault cleared.")

    st.markdown("---")
    st.session_state.auto_refresh = st.checkbox("Auto-refresh Data (1.5s)", value=st.session_state.auto_refresh)
    
    st.markdown("<div class='sidebar-agent'>[SYS] Agent Status: ONLINE<br>[SYS] Model Accuracy: 96.4%<br>[SYS] Last Trained: 2 hrs ago</div>", unsafe_allow_html=True)


# Main Dashboard
st.markdown("<h1 style='text-align: center; color: #E0E0E0; margin-bottom: 25px;'>Agent<span style='color: #00FFC0;'>Ops</span> Command Center</h1>", unsafe_allow_html=True)

# Simulation Step
engine.step()
telemetry_df = engine.get_telemetry_df()
risk_df = detector.predict_risk(telemetry_df)
df = pd.merge(telemetry_df, risk_df, on='machine_id')

# Merge in machine names
machine_info = pd.DataFrame(engine.machines)[['id', 'name']]
df = df.merge(machine_info, left_on='machine_id', right_on='id')

# Top row metrics
total_critical = len(df[df['risk_level'] == 'Critical'])
total_elevated = len(df[df['risk_level'] == 'Elevated'])
avg_temp = df['temperature'].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("System Status", "CRITICAL" if total_critical > 0 else "NOMINAL", delta="-1" if total_critical > 0 else "0", delta_color="inverse")
col2.metric("Critical Alerts", total_critical, delta=total_critical - 0, delta_color="inverse")  
col3.metric("Avg Plant Temp", f"{avg_temp:.1f} °C", f"{(avg_temp - 50):.1f} °C", delta_color="inverse")
col4.metric("Agent Status", "Active", "Real-time")

st.markdown("---")

col_graph, col_data = st.columns([1.5, 1])

with col_graph:
    st.subheader("System Topology & Risk Propagation")
    # Draw Network Graph with Plotly
    G = engine.graph
    
    # We use a custom layout for a horizontal line to look like a manufacturing line
    pos = {
        'M1': (0, 0),
        'M2': (1, 0),
        'M3': (2, 0),
        'M4': (3, 0),
        'M5': (4, 0),
    }
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=6, color='#4A5568'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_color = []
    node_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        m_row = df[df['machine_id'] == node].iloc[0]
        risk = m_row['risk_level']
        if risk == 'Critical':
            node_color.append('#FF3B30') # iOS red
        elif risk == 'Elevated':
            node_color.append('#FF9500') # iOS orange
        else:
            node_color.append('#34C759') # iOS green
            
        node_text.append(f"{m_row['name']}<br>Risk: {m_row['failure_probability']:.1%}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[t.split('<br>')[0] for t in node_text],
        textposition="top center",
        hovertext=node_text,
        textfont=dict(color="white", size=16, family="Inter, sans-serif"),
        marker=dict(
            showscale=False,
            color=node_color,
            size=65,
            line_width=3,
            line_color='#1A202C'
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0,l=20,r=20,t=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 4.5]),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 1]))
                )
    st.plotly_chart(fig, use_container_width=True)

with col_data:
    st.subheader("Agent Diagnostic Feed")
    
    critical_df = df[df['risk_level'] == 'Critical'].copy()
    if not critical_df.empty:
        for _, row in critical_df.iterrows():
            diag_text = detector.generate_diagnostic(row, row['failure_probability'])
            st.error(f"**🚨 ALERT: {row['name']} ({row['machine_id']})**\n\n{diag_text}")
    else:
        elevated_df = df[df['risk_level'] == 'Elevated'].copy()
        if not elevated_df.empty:
            for _, row in elevated_df.iterrows():
                diag_text = detector.generate_diagnostic(row, row['failure_probability'])
                st.warning(f"**⚠️ WARNING: {row['name']} ({row['machine_id']})**\n\n{diag_text}")
        else:
            st.success("**🛡️ AGENT STATUS**\n\nAll systems operating normally. No maintenance recommended at this time.")

    st.markdown("### Live Telemetry")
    # Format dataframe for display
    display_df = df[['id', 'name', 'temperature', 'vibration', 'risk_level']].copy()
    display_df['temperature'] = display_df['temperature'].map('{:.1f} °C'.format)
    display_df['vibration'] = display_df['vibration'].map('{:.2f} Hz'.format)
    
    def highlight_risk(row):
        if row['risk_level'] == 'Critical':
            return ['background-color: rgba(255, 59, 48, 0.2)'] * len(row)
        elif row['risk_level'] == 'Elevated':
            return ['background-color: rgba(255, 149, 0, 0.2)'] * len(row)
        return [''] * len(row)

    st.dataframe(display_df.style.apply(highlight_risk, axis=1), use_container_width=True, hide_index=True)

if st.session_state.auto_refresh:
    time.sleep(1.5)
    st.rerun()
