import numpy as np
import networkx as nx # type: ignore
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class NeuralNetwork:
    def __init__(self, layer_sizes, activation='sigmoid', learning_rate=0.1):
        """
        Initialize the neural network with specified layer sizes.
        
        Parameters:
        - layer_sizes: List of integers, where each integer represents the number of neurons in a layer.
                      The first element is the input layer size, the last is the output layer size.
        - activation: Activation function to use ('sigmoid' or 'relu')
        - learning_rate: Learning rate for gradient descent
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.learning_rate = learning_rate
        self.activation = activation
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(1, self.num_layers):
            # Initialize weights with small random values
            # shape: (neurons in current layer, neurons in previous layer)
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i-1]) * 0.1)
            
            # Initialize biases with small random values
            # shape: (neurons in current layer, 1)
            self.biases.append(np.random.randn(layer_sizes[i], 1) * 0.1)
        
        # For visualization and tracking
        self.history = {
            'loss': [],
            'accuracy': [],
            'weights': [],
            'biases': []
        }
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1.0 / (1.0 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        """Derivative of sigmoid function"""
        sigmoid_z = self.sigmoid(z)
        return sigmoid_z * (1 - sigmoid_z)
    
    def relu(self, z):
        """ReLU activation function"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """Derivative of ReLU function"""
        return np.where(z > 0, 1, 0)
    
    def activate(self, z):
        """Apply selected activation function"""
        if self.activation == 'sigmoid':
            return self.sigmoid(z)
        else:  # 'relu'
            return self.relu(z)
    
    def activate_derivative(self, z):
        """Apply derivative of selected activation function"""
        if self.activation == 'sigmoid':
            return self.sigmoid_derivative(z)
        else:  # 'relu'
            return self.relu_derivative(z)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters:
        - x: Input data, shape (input_size, batch_size)
        
        Returns:
        - activations: List of activations for each layer
        - zs: List of weighted inputs for each layer
        """
        activation = x
        activations = [x]  # List to store all activations, layer by layer
        zs = []  # List to store all z vectors, layer by layer
        
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activate(z)
            activations.append(activation)
        
        return activations, zs
    
    def backward(self, x, y):
        """
        Backward pass (backpropagation) to compute gradients.
        
        Parameters:
        - x: Input data, shape (input_size, batch_size)
        - y: True labels, shape (output_size, batch_size)
        
        Returns:
        - nabla_w: Gradient of the cost function with respect to weights
        - nabla_b: Gradient of the cost function with respect to biases
        """
        # Initialize arrays to store gradients
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        
        # Forward pass
        activations, zs = self.forward(x)
        
        # Backward pass
        # Calculate the error in the output layer
        delta = self.cost_derivative(activations[-1], y) * self.activate_derivative(zs[-1])
        
        # Set the gradients for the output layer
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        
        # Backpropagate the error through the network
        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l+1].T, delta) * self.activate_derivative(zs[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)
        
        return nabla_w, nabla_b
    
    def cost_derivative(self, output_activation, y):
        """
        Derivative of the cost function with respect to the output activation.
        For mean squared error, this is just the difference.
        
        Parameters:
        - output_activation: Activation of the output layer
        - y: True labels
        
        Returns:
        - Derivative of the cost with respect to the output activation
        """
        return output_activation - y
    
    def train(self, training_data, epochs, batch_size, test_data=None):
        """
        Train the neural network using mini-batch stochastic gradient descent.
        
        Parameters:
        - training_data: List of tuples (x, y) with training inputs and labels
        - epochs: Number of epochs to train
        - batch_size: Size of mini-batches
        - test_data: Optional list of tuples (x, y) for evaluating during training
        
        Returns:
        - history: Dictionary with loss and accuracy history
        """
        n = len(training_data)
        
        for epoch in range(epochs):
            # Shuffle the training data
            np.random.shuffle(training_data)
            
            # Create mini-batches
            mini_batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
            
            # Train on each mini-batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch)
            
            # Evaluate and store metrics
            if test_data:
                accuracy = self.evaluate(test_data)
                self.history['accuracy'].append(accuracy)
                print(f"Epoch {epoch}: {accuracy * 100:.2f}% accuracy")
            else:
                print(f"Epoch {epoch} complete")
            
            # Calculate and store loss
            loss = self.calculate_loss(training_data)
            self.history['loss'].append(loss)
            
            # Store weights and biases for visualization
            self.history['weights'].append([w.copy() for w in self.weights])
            self.history['biases'].append([b.copy() for b in self.biases])
    
    def update_mini_batch(self, mini_batch):
        """
        Update the network's weights and biases by applying gradient descent
        using backpropagation to a single mini batch.
        
        Parameters:
        - mini_batch: List of tuples (x, y) with training inputs and labels
        """
        # Initialize gradient accumulators
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        
        # Accumulate gradients for the mini-batch
        for x, y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backward(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        
        # Update weights and biases using averaged gradients
        batch_size = len(mini_batch)
        self.weights = [w - (self.learning_rate / batch_size) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (self.learning_rate / batch_size) * nb for b, nb in zip(self.biases, nabla_b)]
    
    def evaluate(self, test_data):
        """
        Evaluate the network's performance on test data.
        
        Parameters:
        - test_data: List of tuples (x, y) with test inputs and labels
        
        Returns:
        - accuracy: Proportion of correctly classified examples
        """
        test_results = [(np.argmax(self.predict(x)), np.argmax(y)) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results) / len(test_data)
    
    def predict(self, x):
        """
        Make a prediction for a single input.
        
        Parameters:
        - x: Input data, shape (input_size, 1)
        
        Returns:
        - output: Network's prediction, shape (output_size, 1)
        """
        activations, _ = self.forward(x)
        return activations[-1]
    
    def calculate_loss(self, data):
        """
        Calculate the mean squared error loss on data.
        
        Parameters:
        - data: List of tuples (x, y) with inputs and labels
        
        Returns:
        - loss: Mean squared error
        """
        loss = 0
        for x, y in data:
            activation = self.predict(x)
            loss += np.sum((activation - y) ** 2) / len(y)
        return loss / len(data)


# Example: XOR problem implementation
def main():
    # Define XOR problem
    X = np.array([
        [[0], [0]],  # [0, 0]
        [[0], [1]],  # [0, 1]
        [[1], [0]],  # [1, 0]
        [[1], [1]]   # [1, 1]
    ])
    
    # XOR outputs: 0 XOR 0 = 0, 0 XOR 1 = 1, 1 XOR 0 = 1, 1 XOR 1 = 0
    y = np.array([
        [[0]],  # 0 XOR 0 = 0
        [[1]],  # 0 XOR 1 = 1
        [[1]],  # 1 XOR 0 = 1
        [[0]]   # 1 XOR 1 = 0
    ])
    
    # Create training data
    training_data = [(X[i], y[i]) for i in range(4)]
    
    # Create and train neural network
    # XOR needs at least one hidden layer - it's not linearly separable
    nn = NeuralNetwork([2, 4, 1], activation='sigmoid', learning_rate=0.5)
    nn.train(training_data, epochs=5000, batch_size=4)
    
    # Test the network
    for i in range(4):
        prediction = nn.predict(X[i])
        print(f"Input: {X[i].T[0]}, Target: {y[i][0][0]}, Prediction: {prediction[0][0]:.4f}")
    
    # Visualize decision boundary
    visualize_decision_boundary(nn)
    
    # Visualize training progress
    visualize_training_progress(nn)

    # Visualize network architecture
    visualize_network(nn)

def visualize_decision_boundary(nn):
    """
    Visualize the decision boundary of the neural network for the XOR problem.
    
    Parameters:
    - nn: Trained neural network
    """
    # Create a grid of points
    resolution = 100
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Calculate predictions for each point
    Z = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            # Correct input shape: (2, 1)
            input_data = np.array([[X[i, j]], [Y[i, j]]]).reshape(2, 1)
            Z[i, j] = nn.predict(input_data)
    
    # Rest of your plotting code remains the same...
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, cmap='coolwarm', alpha=0.8)
    plt.colorbar(label='Prediction')
    
    # Plot the training points
    colors = ['blue', 'red', 'red', 'blue']
    markers = ['o', 'o', 'o', 'o']
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    
    for i, (input_val, color, marker) in enumerate(zip(inputs, colors, markers)):
        plt.scatter(input_val[0], input_val[1], c=color, marker=marker, s=100, edgecolor='black')
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title('Neural Network Decision Boundary (XOR Problem)')
    plt.grid(True)
    plt.savefig('xor_decision_boundary.png')
    plt.show()


def visualize_training_progress(nn):
    """
    Visualize the training progress of the neural network.
    
    Parameters:
    - nn: Trained neural network with history
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(nn.history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    
    # Create weight evolution animation
    if len(nn.history['weights']) > 50:  # If we have many epochs, sample them
        sample_indices = np.linspace(0, len(nn.history['weights'])-1, 50, dtype=int)
        weight_history = [nn.history['weights'][i] for i in sample_indices]
    else:
        weight_history = nn.history['weights']
    
    # Plot final weights for each layer
    plt.subplot(1, 2, 2)
    for i, w in enumerate(nn.weights):
        plt.bar(range(w.size), w.flatten(), alpha=0.7, label=f'Layer {i+1}')
    
    plt.xlabel('Weight Index')
    plt.ylabel('Weight Value')
    plt.title('Final Weights')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.show()

def visualize_network(nn):
    plt.figure(figsize=(12, 8))
    
    G = nx.DiGraph()
    layer_sizes = nn.layer_sizes
    
    # Add nodes
    for layer_idx, size in enumerate(layer_sizes):
        for node_idx in range(size):
            G.add_node((layer_idx, node_idx))
    
    # Add edges with weights
    edge_colors = []
    edge_widths = []
    for layer_idx, (weights, _) in enumerate(zip(nn.weights, nn.biases)):
        for src in range(weights.shape[1]):
            for dst in range(weights.shape[0]):
                weight = weights[dst, src]
                G.add_edge((layer_idx, src), (layer_idx+1, dst), weight=weight)
                edge_colors.append(weight)
                edge_widths.append(np.abs(weight)*3)
    
    # Position nodes
    pos = {}
    for layer_idx, size in enumerate(layer_sizes):
        layer_pos = np.linspace(0, 1, size)
        for node_idx in range(size):
            pos[(layer_idx, node_idx)] = (layer_idx, layer_pos[node_idx])
    
    # Node colors by layer
    node_colors = []
    for node in G.nodes():
        layer = node[0]
        if layer == 0:
            node_colors.append('skyblue')  # Input layer
        elif layer == len(layer_sizes)-1:
            node_colors.append('lightgreen')  # Output layer
        else:
            node_colors.append('salmon')  # Hidden layers
    
    # Create axis explicitly
    ax = plt.gca()
    
    # Draw everything
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=800,
        node_color=node_colors,
        edgecolors='black',
        linewidths=1,
        ax=ax
    )
    
    edges = nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        edge_cmap=plt.cm.coolwarm,
        width=edge_widths,
        arrowsize=20,
        arrowstyle='->',
        ax=ax
    )
    
    nx.draw_networkx_labels(
        G, pos,
        labels={node: f"{node[0]}.{node[1]}" for node in G.nodes()},
        font_size=10,
        ax=ax
    )   

    # Draw edge labels with weights
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
    
    # Add colorbar with reference to the edges
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, 
                             norm=Normalize(vmin=min(edge_colors), 
                                          vmax=max(edge_colors)))
    sm.set_array(edge_colors)
    plt.colorbar(sm, ax=ax, label='Weight Value', shrink=0.8)
    
    plt.title(f"Neural Network Architecture\n{'-'.join(map(str, layer_sizes))}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('network_visualization.png')
    plt.show()

if __name__ == "__main__":
    main()