import numpy as np

class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, iterations=10000):
        self.input_size = input_size #jml fitur pd input
        self.hidden_size = hidden_size #jml neuron pd hiddden layer
        self.output_size = output_size #jml neuron lapisan output
        self.learning_rate = learning_rate 
        self.iterations = iterations
        
        # inisiasi weight dan bias
        self.weights_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        
        self.weights_output = np.random.rand(self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))
        
    def activation_function(self, z):
        #sigmoid
        return 1 / (1 + np.exp(-z))
    
    def derivative_activation_function(self, output):
        
        return output * (1 - output)
    
    def forward(self, X):
        # (forward pass). Menghitung input dan output dari lapisan tersembunyi dan output.
        self.hidden_input = np.dot(X, self.weights_hidden) + self.bias_hidden
        
        self.hidden_output = self.activation_function(self.hidden_input)
        
        self.output_input = np.dot(self.hidden_output, self.weights_output) + self.bias_output
        self.output = self.activation_function(self.output_input)
        
        return self.output
    
    def backward(self, X, target):
        # (backward pass) untuk memperbarui bobot dan bias berdasarkan gradien kesalahan. Ini melibatkan perhitungan gradien di lapisan output dan penggunaan gradien tersebut untuk menghitung gradien di lapisan tersembuny
        error = target - self.output
        
        # Output layer gradients
        output_delta = error * self.derivative_activation_function(self.output)
        self.weights_output += self.learning_rate * np.dot(self.hidden_output.T, output_delta)
        self.bias_output += self.learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        
        # Hidden layer gradients
        hidden_error = np.dot(output_delta, self.weights_output.T)
        hidden_delta = hidden_error * self.derivative_activation_function(self.hidden_output)
        self.weights_hidden += self.learning_rate * np.dot(X.T, hidden_delta)
        self.bias_hidden += self.learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)
        
    def train(self, X, target):
        for _ in range(self.iterations):
            # Forward pass
            output = self.forward(X)
            
            # Backward pass
            self.backward(X, target)
            #Melatih model dengan melakukan iterasi forward dan backward.
    def predict(self, X):
        # Menggunakan model yang telah dilatih untuk membuat prediksi baru
        return self.forward(X)

