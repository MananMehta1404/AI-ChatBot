import torch.nn as nn

# Creating the Neural Network Model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        """
        Constructor for the NeuralNet class.

        Parameters:
            input_size (int): The size of the input layer.
            hidden_size (int): The size of the hidden layer.
            num_classes (int): The number of output classes.

        Returns: None
        """

        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)    # Input layer
        self.l2 = nn.Linear(hidden_size, hidden_size)   # Hidden layer
        self.l3 = nn.Linear(hidden_size, num_classes)   # Output layer
        self.relu = nn.ReLU()                           # Activation function
    
    def forward(self, x):
        """
        Performs the forward pass through the neural network.

        Parameters:
            x: The input tensor.

        Returns:
            out: The output tensor after passing through the network layers.
        """

        out = self.l1(x)            # Pass the input tensor through the first layer
        out = self.relu(out)        # Apply the ReLU activation function
        out = self.l2(out)          # Pass the output tensor through the second layer
        out = self.relu(out)        # Apply the ReLU activation function
        out = self.l3(out)          # Pass the output tensor through the third layer

        # No Activation and No Softmax at the end
        return out