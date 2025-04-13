# XOR NN 🧠⚡

**A Minimalist Neural Network Implementation for XOR Problem with Visualizations**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0-lightgrey?logo=flask)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📦 Project Structure
```
NeuralXOR/
├── app.py # Flask web interface
├── BasicNN.py # Core neural network class
├── generate.py # Visualization generator
├── UnderstandingNNs.pdf # Theoretical background
├── templates/ # Web UI templates
│ └── index.html
└── visualisations/ # Generated plots
  └── decision_boundary.png
  └──network_graph.png
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
```python
[Input(2)] → [Hidden(4), sigmoid] → [Output(1), sigmoid]
Training Parameters
Parameter    	   Value	     Description
Learning Rate	   0.5	       Controls weight update step size
Loss Function	   MSE	       Mean Squared Error
Batch Size	     4	         Full-batch training for XOR
Epochs	         5000	     Typical training iterations needed
```
##  🛠 Development
Dependencies
```
numpy>=1.21.0
matplotlib>=3.5.0
flask>=2.0.0
networkx>=2.6.0
```

Authored by -  Karan Sehgal

