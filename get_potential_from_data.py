import torch
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class PotentialfromData():
    '''
    data: tensor, shape (N,2)
    '''
    def __init__(self, data, f):
        self.x = data[:,0]
        self.y = data[:,1]
        self.N = len(data)
        self.f = f

    def cumulative_distances(self):
        distances = torch.sqrt(torch.pow(torch.diff(self.x), 2) + torch.pow(torch.diff(self.y), 2))
        
        return torch.cat([torch.tensor([0.]), torch.cumsum(distances, dim=0)])
    
    def get_gamma(self, qty):
        minimum = 0
        maximum = torch.max(self.cumulative_distances())

        return torch.linspace(minimum, maximum, int(qty), requires_grad=True)

    def tangent_vector_normalized(self):
        gamma = self.get_gamma(self.N)[:,None]
        outputs  = self.f(gamma)

        x = outputs[:,0]
        y = outputs[:,1]

        x_grads = grad(outputs=x, inputs=gamma, grad_outputs=torch.ones_like(x), 
                       retain_graph=True)[0]
        y_grads = grad(outputs=y, inputs=gamma, grad_outputs=torch.ones_like(y), 
                       retain_graph=True)[0]
        
        magnitude = torch.sqrt(x_grads**2 + y_grads**2)

        return x_grads/magnitude, y_grads/magnitude
    
    def curvature_vector_normalized(self):
        gamma   = self.get_gamma(self.N)[:,None]
        outputs = self.f(gamma)

        x = outputs[:,0]
        y = outputs[:,1]

        # First derivatives
        x_grads = grad(outputs=x, inputs=gamma, grad_outputs=torch.ones_like(x), 
                       create_graph=True)[0]
        y_grads = grad(outputs=y, inputs=gamma, grad_outputs=torch.ones_like(y), 
                       create_graph=True)[0]

        # Second derivatives (using the original first derivatives without detaching)
        x_ggrads = grad(outputs=x_grads, inputs=gamma, grad_outputs=torch.ones_like(x_grads), 
                        create_graph=True)[0]
        y_ggrads = grad(outputs=y_grads, inputs=gamma, grad_outputs=torch.ones_like(y_grads), 
                        create_graph=True)[0]

        magnitude = torch.sqrt(x_ggrads**2 + y_ggrads**2)

        return x_ggrads/magnitude, y_ggrads/magnitude
    
