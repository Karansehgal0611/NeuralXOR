from flask import Flask, render_template, request, jsonify # type: ignore
import numpy as np
import json
import math

app = Flask(__name__)

class NeuralNetwork:
    def __init__(self, layer_sizes, activation='sigmoid', learning_rate=0.1):
        """
        Initialize neural network with specified layer sizes
        
        Parameters:
        - layer_sizes: List of integers representing neurons in each layer
        - activation: 'sigmoid' or 'relu'
        - learning_rate: Learning rate for gradient descent
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation = activation
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            # Initialize weights between -1 and 1
            layer_weights = []
            for j in range(layer_sizes[i+1]):
                neuron_weights = []
                for k in range(layer_sizes[i]):
                    neuron_weights.append(np.random.random() * 2 - 1)
                layer_weights.append(neuron_weights)
            self.weights.append(layer_weights)
            
            # Initialize biases between -1 and 1
            layer_biases = []
            for j in range(layer_sizes[i+1]):
                layer_biases.append(np.random.random() * 2 - 1)
            self.biases.append(layer_biases)
        
        self.epoch = 0
        self.loss = 0
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + math.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        sig_x = self.sigmoid(x)
        return sig_x * (1 - sig_x)
    
    def relu(self, x):
        """ReLU activation function"""
        return max(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU function"""
        return 1 if x > 0 else 0
    
    def activate(self, x):
        """Apply selected activation function"""
        return self.sigmoid(x) if self.activation == 'sigmoid' else self.relu(x)
    
    def activate_derivative(self, x):
        """Apply derivative of selected activation function"""
        return self.sigmoid_derivative(x) if self.activation == 'sigmoid' else self.relu_derivative(x)
    
    def forward_pass(self, inputs):
        """
        Forward pass through the network
        
        Parameters:
        - inputs: Input data
        
        Returns:
        - Dictionary with activations and zs
        """
        current_activations = list(inputs)
        activations = [current_activations]
        zs = []
        
        for i in range(len(self.weights)):
            layer_zs = []
            layer_activations = []
            
            for j in range(len(self.weights[i])):
                z = self.biases[i][j]
                for k in range(len(self.weights[i][j])):
                    z += self.weights[i][j][k] * current_activations[k]
                layer_zs.append(z)
                layer_activations.append(self.activate(z))
            
            zs.append(layer_zs)
            current_activations = layer_activations
            activations.append(current_activations)
        
        return {"activations": activations, "zs": zs}
    
    def backpropagation(self, x, y):
        """
        Backward pass to compute gradients
        
        Parameters:
        - x: Input data
        - y: Target output
        
        Returns:
        - Dictionary with weight and bias gradients
        """
        # Forward pass
        result = self.forward_pass(x)
        activations = result["activations"]
        zs = result["zs"]
        
        # Initialize arrays to store gradients
        nabla_w = []
        for layer in self.weights:
            nabla_w_layer = []
            for neuron in layer:
                nabla_w_neuron = [0] * len(neuron)
                nabla_w_layer.append(nabla_w_neuron)
            nabla_w.append(nabla_w_layer)
        
        nabla_b = []
        for layer in self.biases:
            nabla_b.append([0] * len(layer))
        
        # Backward pass
        # Calculate the error in the output layer
        delta = []
        output_errors = []
        for i in range(len(activations[-1])):
            output_errors.append(activations[-1][i] - y[i])
        
        # Calculate delta for the output layer
        for i in range(len(output_errors)):
            delta.append(output_errors[i] * self.activate_derivative(zs[-1][i]))
        
        # Set the gradients for the output layer
        for i in range(len(nabla_b[-1])):
            nabla_b[-1][i] = delta[i]
            for j in range(len(nabla_w[-1][i])):
                nabla_w[-1][i][j] = delta[i] * activations[-2][j]
        
        # Backpropagate the error through the network
        for l in range(len(self.weights) - 2, -1, -1):
            new_delta = []
            
            for i in range(len(self.weights[l])):
                error = 0
                for j in range(len(self.weights[l+1])):
                    error += self.weights[l+1][j][i] * delta[j]
                new_delta.append(error * self.activate_derivative(zs[l][i]))
            
            delta = new_delta
            
            # Set the gradients for this layer
            for i in range(len(nabla_b[l])):
                nabla_b[l][i] = delta[i]
                for j in range(len(nabla_w[l][i])):
                    nabla_w[l][i][j] = delta[i] * activations[l][j]
        
        return {"nabla_w": nabla_w, "nabla_b": nabla_b}
    
    def update_parameters(self, nabla_w, nabla_b):
        """
        Update network parameters with calculated gradients
        
        Parameters:
        - nabla_w: Weight gradients
        - nabla_b: Bias gradients
        """
        # Update weights
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] -= self.learning_rate * nabla_w[i][j][k]
        
        # Update biases
        for i in range(len(self.biases)):
            for j in range(len(self.biases[i])):
                self.biases[i][j] -= self.learning_rate * nabla_b[i][j]
    
    def train(self, data_points, batch_size=10, num_epochs=50):
        """
        Train the neural network
        
        Parameters:
        - data_points: Training data
        - batch_size: Size of mini-batches
        - num_epochs: Number of training epochs
        
        Returns:
        - Dictionary with loss and epoch
        """
        current_epoch = 0
        
        while current_epoch < num_epochs:
            # Mini-batch gradient descent
            batch_nabla_w = []
            for layer in self.weights:
                batch_nabla_w_layer = []
                for neuron in layer:
                    batch_nabla_w_neuron = [0] * len(neuron)
                    batch_nabla_w_layer.append(batch_nabla_w_neuron)
                batch_nabla_w.append(batch_nabla_w_layer)
            
            batch_nabla_b = []
            for layer in self.biases:
                batch_nabla_b.append([0] * len(layer))
            
            # Process random examples per batch
            for i in range(batch_size):
                random_index = np.random.randint(0, len(data_points))
                point = data_points[random_index]
                
                result = self.backpropagation([point["x"], point["y"]], [point["z"]])
                nabla_w = result["nabla_w"]
                nabla_b = result["nabla_b"]
                
                # Accumulate gradients
                for l in range(len(batch_nabla_w)):
                    for j in range(len(batch_nabla_w[l])):
                        for k in range(len(batch_nabla_w[l][j])):
                            batch_nabla_w[l][j][k] += nabla_w[l][j][k]
                
                for l in range(len(batch_nabla_b)):
                    for j in range(len(batch_nabla_b[l])):
                        batch_nabla_b[l][j] += nabla_b[l][j]
            
            # Update parameters with averaged gradients
            for l in range(len(batch_nabla_w)):
                for j in range(len(batch_nabla_w[l])):
                    for k in range(len(batch_nabla_w[l][j])):
                        batch_nabla_w[l][j][k] /= batch_size
            
            for l in range(len(batch_nabla_b)):
                for j in range(len(batch_nabla_b[l])):
                    batch_nabla_b[l][j] /= batch_size
            
            self.update_parameters(batch_nabla_w, batch_nabla_b)
            
            # Calculate loss over the entire dataset
            total_loss = 0
            for point in data_points:
                prediction = self.forward_pass([point["x"], point["y"]])["activations"][-1][0]
                total_loss += (prediction - point["z"]) ** 2
            
            total_loss /= len(data_points)
            self.loss = total_loss
            
            current_epoch += 1
            self.epoch = current_epoch
            
            # Stop if loss is small enough
            if total_loss < 0.01:
                break
        
        return {"loss": self.loss, "epoch": self.epoch}

# Initialize neural network
nn = NeuralNetwork([2, 3, 1], activation='sigmoid', learning_rate=0.1)

# Generate XOR-like data
def generate_data():
    data = []
    for i in range(100):
        x = np.random.random()
        y = np.random.random()
        # XOR function: (x > 0.5 and y < 0.5) or (x < 0.5 and y > 0.5)
        z = 1 if (x > 0.5 and y < 0.5) or (x < 0.5 and y > 0.5) else 0
        data.append({"x": x, "y": y, "z": z})
    return data

data_points = generate_data()

# Create decision boundary grid
def create_decision_grid(resolution=40):
    grid_points = []
    for i in range(resolution):
        for j in range(resolution):
            x = i / (resolution - 1)
            y = j / (resolution - 1)
            result = nn.forward_pass([x, y])
            prediction = result["activations"][-1][0]
            grid_points.append({"x": x, "y": y, "prediction": prediction})
    return grid_points

@app.route('/')
def index():
    return render_template('index.html', 
                          nn_layers=nn.layer_sizes, 
                          activation=nn.activation,
                          learning_rate=nn.learning_rate)

@app.route('/api/data')
def get_data():
    return jsonify(data_points)

@app.route('/api/grid')
def get_grid():
    grid = create_decision_grid()
    return jsonify(grid)

@app.route('/api/network')
def get_network():
    return jsonify({
        "weights": nn.weights,
        "biases": nn.biases,
        "layers": nn.layer_sizes,
        "epoch": nn.epoch,
        "loss": nn.loss
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    inputs = [data.get('x1', 0.5), data.get('x2', 0.5)]
    result = nn.forward_pass(inputs)
    output = result["activations"][-1][0]
    return jsonify({"output": output})

@app.route('/api/train', methods=['POST'])
def train():
    data = request.get_json()
    activation_function = data.get('activation', 'sigmoid')
    learning_rate = float(data.get('learning_rate', 0.1))
    
    # Update network parameters if needed
    if nn.activation != activation_function or nn.learning_rate != learning_rate:
        nn.activation = activation_function
        nn.learning_rate = learning_rate
    
    # Train the network
    result = nn.train(data_points)
    return jsonify({
        "loss": result["loss"],
        "epoch": result["epoch"]
    })

@app.route('/api/reset', methods=['POST'])
def reset():
    global nn
    nn = NeuralNetwork(nn.layer_sizes, nn.activation, nn.learning_rate)
    return jsonify({"status": "reset successful"})

if __name__ == '__main__':
    app.run(debug=True)