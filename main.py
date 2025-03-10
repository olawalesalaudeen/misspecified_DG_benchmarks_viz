# main.py

import streamlit as st
from simulation_mode.simulation import run_simulation
from plotting_mode.plotting import run_real_data_plotting

# Sidebar navigation
st.sidebar.title("Are Domain Generalization Benchmarks with Accuracy on the Line Misspecified?")
selection = st.sidebar.radio("Playground Options:", ["Simulation Mode", "Real Data Mode"])

if selection == "Simulation Mode":
    run_simulation()
elif selection == "Real Data Mode":
    run_real_data_plotting()
