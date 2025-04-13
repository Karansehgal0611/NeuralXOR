from reportlab.lib.pagesizes import A4 # type: ignore
from reportlab.pdfgen import canvas # type: ignore
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer # type: ignore
from reportlab.lib.styles import getSampleStyleSheet # type: ignore

# File name for the PDF
file_name = "Understanding_Neural_Networks.pdf"

# Create a SimpleDocTemplate instance
doc = SimpleDocTemplate(file_name, pagesize=A4)

# Get sample styles for formatting
styles = getSampleStyleSheet()
story = []

# Add title
title = Paragraph("Understanding Neural Networks from Scratch", styles['Title'])
story.append(title)
story.append(Spacer(1, 12))

# Add content sections
content_sections = [
    ("Introduction", """
    Neural networks are computational models inspired by the human brain. They consist of connected nodes (neurons) organized in layers that transform input data into predictions. In this guide, we'll break down how neural networks work from the ground up.
    """),
    ("Core Components of a Neural Network", """
    1. Neurons:
       - Receives inputs from previous layers
       - Applies weights to those inputs
       - Sums the weighted inputs and adds a bias
       - Passes the result through an activation function
       - Produces an output that's sent to the next layer

    2. Layers:
       - Input Layer: Receives the initial data
       - Hidden Layers: Intermediate layers that perform computations
       - Output Layer: Produces the final prediction

    3. Weights and Biases:
       - Weights: Determine the strength of connections between neurons
       - Biases: Allow the network to shift the activation function

    4. Activation Functions:
       Activation functions introduce non-linearity, allowing the network to learn complex patterns:
       - Sigmoid: σ(x) = 1 / (1 + e^(-x))
       - ReLU: f(x) = max(0, x)
       - Tanh: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    """),
    ("How Neural Networks Work: Step by Step", """
    1. Forward Propagation:
       - Input Layer: The network receives input data.
       - Hidden Layers: For each neuron in each hidden layer:
         z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b; a = activation(z)
       - Output Layer: The final layer produces the prediction.

    2. Loss Calculation:
       - Mean Squared Error (MSE): MSE = (1/n) Σ(y - ŷ)²
       - Cross-Entropy Loss: -Σ(y log(ŷ))

    3. Backpropagation:
       - Calculate error at output layer.
       - Update weights and biases using gradients.

    4. Gradient Descent:
       - Update weights and biases: w_new = w_old - learning_rate * gradient
    """),
    ("XOR Problem: A Classic Example", """
    The XOR logic problem demonstrates why hidden layers are needed for non-linearly separable data.
    XOR Logic:
      0 XOR 0 = 0
      0 XOR 1 = 1
      1 XOR 0 = 1
      1 XOR 1 = 0
    """),
    ("Key Terminology", """
    Key terms include Epoch, Batch Size, Learning Rate, Overfitting, Regularization.
    """),
    ("Types of Neural Networks", """
    Types include Feedforward Neural Networks, CNNs, RNNs, LSTMs, Transformers.
    """),
    ("Conclusion", """
    Neural networks are powerful tools for pattern recognition and prediction tasks.
    """)
]

# Add content sections to story
for section_title, section_content in content_sections:
    story.append(Paragraph(section_title, styles['Heading2']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(section_content.strip(), styles['BodyText']))
    story.append(Spacer(1, 12))

# Build PDF document
doc.build(story)

print(f"PDF '{file_name}' has been successfully created.")
