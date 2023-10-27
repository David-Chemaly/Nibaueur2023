import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot
from Dataset_Network import BasicNN, NotSoBasicNN, CustomDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


DTYPE = torch.float32

class gamma_to_XY():
    '''
    data: tensor shape (N,2)
    '''
    def __init__(self,data):
        self.x = data[:,0]
        self.y = data[:,1]
        self.N = len(data)

    def cumulative_distances(self):
        distances = torch.sqrt(torch.pow(torch.diff(self.x), 2) + torch.pow(torch.diff(self.y), 2))
        
        return torch.cat([torch.tensor([0.]), torch.cumsum(distances, dim=0)])
    
    def get_gamma(self, qty):
        minimum = 0
        maximum = torch.max(self.cumulative_distances())

        return torch.linspace(minimum, maximum, int(qty))
    
    def linear_interpolation(self, gamma):

        cumul_distance = self.cumulative_distances()

        # Find the indices of the two data points that 'a' lies between
        idx = torch.searchsorted(cumul_distance, gamma, right=True) - 1
        idx = torch.clamp(idx, 0, len(cumul_distance) - 2)  # ensure index is within bounds
        
        # Compute the weights for interpolation
        fraction = (gamma - cumul_distance[idx]) / (cumul_distance[idx + 1] - cumul_distance[idx])
        
        # Linearly interpolate the x and y values
        x_val = self.x[idx] + fraction * (self.x[idx + 1] - self.x[idx])
        y_val = self.y[idx] + fraction * (self.y[idx + 1] - self.y[idx])
        
        return x_val, y_val
    
    def NN_interpolation(self, qty, bs, lr, e, model_type, save_name):
        # more_gamma = self.get_gamma(qty)[:,None]
        # more_x, more_y = self.linear_interpolation(more_gamma)
        # more_labels = torch.cat((more_x,more_y), dim=1).to(DTYPE)

        more_gamma = self.get_gamma(self.N)[:,None]
        more_labels = torch.cat((self.x[:,None],self.y[:,None]), dim=1).to(DTYPE)

        dataset    = CustomDataset(more_gamma, more_labels)
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

        # Initialize model, criterion, and optimizer
        if model_type == 'Basic':
            model     = BasicNN()
        elif model_type == 'Not So Basic':
            model     = NotSoBasicNN()
        else:
            print('No Valid Model Type')
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)

        # Training loop
        for epoch in tqdm(range(e), leave=True):
            for batch_data, batch_labels in dataloader:
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_data)
                
                # Calculate loss
                loss = criterion(outputs, batch_labels)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
        torch.save(model.state_dict(), save_name+'.pth')





