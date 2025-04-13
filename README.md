# XOR NN 🧠⚡

**A Minimalist Neural Network Implementation for XOR Problem with Visualizations**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0-lightgrey?logo=flask)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

![XOR Decision Boundary](visualisations/decision_boundary.png)

## 📦 Project Structure
```
BasicNN/
├── app.py # Flask web interface
├── BasicNN.py # Core neural network class
├── generate.py # Visualization generator
├── UnderstandingNNs.pdf # Theoretical background
├── templates/ # Web UI templates
│ └── index.html
└── visualisations/ # Generated plots
├── decision_boundary.png
├── network_graph.png
└── training_progress.png
```

## 🌟 Key Features

- **Pure Python Implementation**  
  No ML frameworks - understand backpropagation from scratch
- **Interactive Web Dashboard**  
  Train and predict via browser interface
- **Multiple Visualization Modes**  
  - Decision boundaries  
  - Network architecture graphs  
  - Training progress plots
- **Educational Focus**  
  Companion PDF explains mathematical foundations

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/Karansehgal0611/NeuralXOR.git
cd NeuralXOR
python -m venv venv
source venv/bin/activate  # Linux/Mac
# {venv\Scripts\activate}     # Windows
```

## 🧠 Neural Network Details

### Architecture
[![Input Nodes](https://img.shields.io/badge/Input_Layer-2_nodes-blue)] 
[![Hidden Nodes](https://img.shields.io/badge/Hidden_Layer-4_nodes-green)]
[![Activation](https://img.shields.io/badge/Activation-Sigmoid-orange)]
[![Output Nodes](https://img.shields.io/badge/Output_Layer-1_node-red)]

### Training Configuration
[![Learning Rate](https://img.shields.io/badge/Learning_Rate-0.5-yellow)] 
[![Loss](https://img.shields.io/badge/Loss-MSE-blueviolet)]
[![Batch Size](https://img.shields.io/badge/Batch_Size-4-ff69b4)]
[![Epochs](https://img.shields.io/badge/Epochs-5000-success)]

##  🛠 Development
Dependencies
```
numpy>=1.21.0
matplotlib>=3.5.0
flask>=2.0.0
networkx>=2.6.0
```

Authored by -  Karan Sehgal

