import torch.nn as nn


# more free version of the MLP 
class MLPClassifier(nn.Module):
    def __init__(self, d_in=784, n_out=10, hidden_units=[784,256], activation='relu', dropout=0.1, batch_norm=False):
        super(MLPClassifier, self).__init__()
        
        layers = []
        in_features = d_in
        
        # Map string to activation function
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
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(units))
            
            layers.append(act_fn)
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            in_features = units
        
        # Output layer
        layers.append(nn.Linear(in_features, n_out))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)