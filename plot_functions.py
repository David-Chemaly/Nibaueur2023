import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import grad
from stream_evolution_potential import stream_evolution

def stream_3projections(start, end, size, slice, pos, vel, acc, potential, output_name):

    # For acceleration vector field
    X, Y, Z = torch.meshgrid(torch.linspace(start,end,size), 
                             torch.linspace(start,end,size),
                             torch.linspace(start,end,size))
    
    if potential == 'Point Mass':
        phi = stream_evolution.point_mass_potential(X.requires_grad_(True),Y.requires_grad_(True),Z.requires_grad_(True))
    elif potential == 'Line':
        phi = stream_evolution.line_potential(X.requires_grad_(True),Y.requires_grad_(True),Z.requires_grad_(True))
    elif potential == 'Disk':
        phi = stream_evolution.disk_potential(X.requires_grad_(True),Y.requires_grad_(True),Z.requires_grad_(True))
    elif potential == 'Halo':
        phi = stream_evolution.halo_potential(X.requires_grad_(True),Y.requires_grad_(True),Z.requires_grad_(True))
    else:
        print('No valide potential')

    a = grad(phi, (X,Y,Z), grad_outputs=torch.ones_like(phi))
    a_x = -a[0]0
    a_y = -a[1]
    a_z = -a[2]

    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.title('X-Y')
    plt.quiver(X[:,:,slice].detach().numpy(),
            Y[:,:,slice].detach().numpy(),
            a_x[:,:,slice].detach().numpy(),
            a_y[:,:,slice].detach().numpy(),
            scale=0.2,
            color='grey')
    plt.quiver(pos[:,0].detach().numpy(),
            pos[:,1].detach().numpy(),
            acc[:,0].detach().numpy(),
            acc[:,1].detach().numpy(),
            scale=0.2,
            color='red')
    plt.plot(pos[:,0],pos[:,1],color='black')
    plt.subplot(1,3,2)
    plt.title('X-Z')
    plt.quiver(X[:,slice,:].detach().numpy(),
            Z[:,slice,:].detach().numpy(),
            a_x[:,slice,:].detach().numpy(),
            a_z[:,slice,:].detach().numpy(),
            scale=0.2,
            color='grey')
    plt.quiver(pos[:,0].detach().numpy(),
            pos[:,2].detach().numpy(),
            acc[:,0].detach().numpy(),
            acc[:,2].detach().numpy(),
            scale=0.2,
            color='red')
    plt.plot(pos[:,0],pos[:,2],color='black')
    plt.subplot(1,3,3)
    plt.title('Y-Z')
    plt.quiver(Y[slice,:,:].detach().numpy(),
            Z[slice,:,:].detach().numpy(),
            a_y[slice,:,:].detach().numpy(),
            a_z[slice,:,:].detach().numpy(),
            scale=0.2,
            color='grey')
    plt.quiver(pos[:,1].detach().numpy(),
            pos[:,2].detach().numpy(),
            acc[:,1].detach().numpy(),
            acc[:,2].detach().numpy(),
            scale=0.2,
            color='red')
    plt.plot(pos[:,1],pos[:,2],color='black')
    plt.savefig(output_name)