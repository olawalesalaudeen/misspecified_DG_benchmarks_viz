# Are Domain Generalization Benchmarks with Accuracy on the Line Misspecified?

This repository accompanies the paper
Salaudeen, Olawale, et al. "Are Domain Generalization Benchmarks with Accuracy on the Line Misspecified?." arXiv preprint arXiv:2504.00186 (2025).

We provide:
1. **Simulations & Plots** that illustrate when and why common OOD benchmarks fail to detect reliance on spurious correlations.
2. A **Streamlit app** (`main.py`) for interactive exploration of accuracy-on-the-line patterns under different simulated shifts: https://misspecified-dg-benchmarks-viz.streamlit.app/

---

## 📖 Paper

Read the full paper on arXiv:
https://arxiv.org/abs/2504.00186

---

## 🚀 Features

- **Simulation Mode** (`/simulation_mode`):
    Generate toy domain splits under controlled shifts and visualize correlation patterns. Users can either mimic their expected spurious-correlation structure to produce and examine “accuracy on the line,” or interactively explore different spurious-correlation structures that yield similar ID vs. OOD accuracy patterns observed in real benchmarks. This application helps user determine if their benchmark or OOD generalization task is misspecified.
- **Plotting Mode** (`/plotting_mode`):
    Load real benchmark data (e.g., PACS, VLCS, Waterbirds) to reproduce “accuracy on the line” plots.
- **Interactive App**:
    Tune parameters (spurious-/domain-general-signal strength, shift severity, etc.) in real time and observe how Pearson R ID vs. OOD changes.

---
## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/olawalesalaudeen/misspecified_DG_benchmarks_viz.git
   cd misspecified_DG_benchmarks_viz

---
## Running the Application

1) Create a virtual environment with `python=3.9` and install packages using `pip install -r requirements.txt`.
2) Run `streamlit run main.py`
