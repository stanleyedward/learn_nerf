import torch
from torch import nn
class Voxels(nn.Module):
    def __init__(self, nb_voxels:int = 100, scale = 1, device = 'cpu'):
        super(Voxels, self).__init__()
        
        self.voxels = torch.nn.Parameter(torch.rand((nb_voxels,nb_voxels, nb_voxels, 4), device = device), requires_grad=True) #colors and density cannot be negative therefore dont use normal distribtion
        self.nb_voxels = nb_voxels
        self.device = device
        self.scale = scale
        
    def forward(self, xyz):
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
        
        condition = (x.abs() < (self.scale / 2)) & (y.abs() < (self.scale / 2)) & (z.abs() < (self.scale / 2))

        colors_and_densities = torch.zeros((xyz.shape[0], 4))
        # colors_and_densities[condition, :3] = torch.Tensor([1., 0., 0.]) #colors #uncomment for red cube
        # colors_and_densities[condition, -1] = 10 #density #uncomment for red cube
        
        idx_x = (x[condition] / (self.scale / self.nb_voxels) + self.nb_voxels / 2).type(torch.long) #comment for red cube
        idx_y = (y[condition] / (self.scale / self.nb_voxels) + self.nb_voxels / 2).type(torch.long) #comment for red cube
        idx_z = (z[condition] / (self.scale / self.nb_voxels) + self.nb_voxels / 2).type(torch.long) #comment for red cube
        
        colors_and_densities[condition, :3] = self.voxels[idx_x, idx_y, idx_z, :3] #colors #comment for red cube
        colors_and_densities[condition, -1] = self.voxels[idx_x, idx_y, idx_z, -1]  # * 10 #density #comment for red cube
        
        return colors_and_densities[:, :3], colors_and_densities[:, -1:] #uncomment for red cube
        # return torch.sigmoid(colors_and_densities[:, :3]), torch.relu(colors_and_densities[:, -1:]) # uncomment if its barely visible
    
    def intersect(self, x):
        return self.forward(x)