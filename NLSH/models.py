import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, d_in, n_out, hidden_units, activation='relu', dropout=0.1, batch_norm=True):
        """
        Args:
            d_in (int): Input dimension (e.g., 784 for MNIST).
            n_out (int): Output dimension (number of partitions m).
            hidden_units (list): List of integers defining hidden layer sizes.
            activation (str): Activation function name.
            dropout (float): Dropout probability.
            batch_norm (bool): Whether to use Batch Normalization.
        """
        super(MLPClassifier, self).__init__()
        
        layers = []
        in_features = d_in
        
        # Activation map
        activations = {
            'relu': nn.ReLU(),
            'leakyrelu': nn.LeakyReLU(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU()
        }
        act_fn = activations.get(activation.lower(), nn.ReLU())
        
        # Build hidden layers
        for units in hidden_units:
            layers.append(nn.Linear(in_features, units))
            
            # Batch Norm is usually applied before activation
            if batch_norm:
                layers.append(nn.BatchNorm1d(units))
            
            layers.append(act_fn)
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            in_features = units
        
        # Output layer (Logits)
        layers.append(nn.Linear(in_features, n_out))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)