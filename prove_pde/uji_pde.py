import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

def physics_loss_ori(model, inputs, g, n):
    #DATA SHAPE
    #timestep x width x height x (x y h u v dzdx dzdy time)
    # Enable gradient computation for PDE terms
    inputs_clone = inputs.clone()
    inputs_clone.requires_grad = True
    #inputs.requires_grad_(True)

    # Forward pass
    outputs = model(inputs_clone)  # [u, v, h]
    h, u, v = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3]

    h = torch.clamp(h, min=1e-6)

    #print('cek u', u)
    #print('cek t', inputs[:, 7])

    u_t = torch.autograd.grad(u.sum(), inputs_clone, create_graph=True)[0][:, 7:8]
    u_x = torch.autograd.grad(u.sum(), inputs_clone, create_graph=True)[0][:, 0:1]
    u_y = torch.autograd.grad(u.sum(), inputs_clone, create_graph=True)[0][:, 1:2]

    v_t = torch.autograd.grad(v.sum(), inputs_clone, create_graph=True)[0][:, 7:8]
    v_x = torch.autograd.grad(v.sum(), inputs_clone, create_graph=True)[0][:, 0:1]
    v_y = torch.autograd.grad(v.sum(), inputs_clone, create_graph=True)[0][:, 1:2]

    h_t = torch.autograd.grad(h.sum(), inputs_clone, create_graph=True)[0][:, 7:8]
    h_x = torch.autograd.grad(h.sum(), inputs_clone, create_graph=True)[0][:, 0:1]
    h_y = torch.autograd.grad(h.sum(), inputs_clone, create_graph=True)[0][:, 1:2]

    S_x = inputs_clone[:, 5]
    S_y = inputs_clone[:, 6]

    S0_x = -S_x
    S0_y = -S_y
    #S0_x = S_x
    #S0_y = S_y

    Sf_x = n**2 * u * torch.sqrt(u**2 + v**2) / h**(4/3)
    Sf_y = n**2 * v * torch.sqrt(u**2 + v**2) / h**(4/3)

    # Momentum equation residuals
    pde_u = u_t + u * u_x + v * u_y + g * h_x - g * S0_x + g * Sf_x
    pde_v = v_t + u * v_x + v * v_y + g * h_y - g * S0_y + g * Sf_y

    # Continuity equation residual
    pde_h = h_t + (h * u_x + u * h_x) + (h * v_y + v * h_y)

    # PDE loss (sum of squared residuals)
    loss_pde = (pde_u**2).mean() + (pde_v**2).mean() + (pde_h**2).mean()
    return loss_pde


# Finite difference with padding for shape alignment
def finite_difference(data, axis, step):
    diff = (data.narrow(axis, 1, data.size(axis) - 1) - data.narrow(axis, 0, data.size(axis) - 1)) / step
    pad_shape = list(data.shape)
    pad_shape[axis] = 1  # Create a single slice padding
    padding = torch.zeros(pad_shape, dtype=data.dtype, device=data.device)
    diff = torch.cat([diff, padding], dim=axis)  # Align shapes by padding at the end
    return diff

def physics_loss2(inputs, g, n):
    # Feature indices for easier access
    IDX_X = 0
    IDX_Y = 1
    IDX_H = 2
    IDX_U = 3
    IDX_V = 4
    IDX_DZDX = 5
    IDX_DZDY = 6
    IDX_TIME = 7

    # Parameters
    dx = 0.2  # Spatial step (x-direction)
    dy = 0.2  # Spatial step (y-direction)
    dt = 1  # Time step
    g = 9.81  # Gravity
    n = 0.03  # Manning's coefficient

    # Extract ground truth data
    h = inputs[:, :, :, IDX_H]
    u = inputs[:, :, :, IDX_U]
    v = inputs[:, :, :, IDX_V]
    dzdx = inputs[:, :, :, IDX_DZDX]
    dzdy = inputs[:, :, :, IDX_DZDY]

    h = torch.clamp(h, min=1e-6)

    #print('cek size', u.size(), v.size())

    # Compute derivatives
    h_t = finite_difference(h, axis=0, step=dt)
    h_x = finite_difference(h, axis=1, step=dx)
    h_y = finite_difference(h, axis=2, step=dy)

    u_t = finite_difference(u, axis=0, step=dt)
    u_x = finite_difference(u, axis=1, step=dx)
    u_y = finite_difference(u, axis=2, step=dy)

    v_t = finite_difference(v, axis=0, step=dt)
    v_x = finite_difference(v, axis=1, step=dx)
    v_y = finite_difference(v, axis=2, step=dy)

    print('cek u', u_t.mean(), u_x.mean(), u_y.mean())

    #print('cek size derivatives', h_t.size(), h_x.size(), h_y.size(), u_t.size(), u_x.size(), u_y.size(), v_t.size(), v_x.size(), v_y.size(), )

    # Source terms
    S0_x = -dzdx
    S0_y = -dzdy
    Sf_x = n**2 * u * torch.sqrt(u**2 + v**2) / h**(4 / 3)
    Sf_y = n**2 * v * torch.sqrt(u**2 + v**2) / h**(4 / 3)



    # Calculate residuals for the PDEs
    pde_u = u_t + u * u_x + v * u_y + g * h_x - g * S0_x + g * Sf_x
    pde_v = v_t + u * v_x + v * v_y + g * h_y - g * S0_y + g * Sf_y
    pde_h = h_t + (h * u_x + u * h_x) + (h * v_y + v * h_y)

    # Compute the PDE loss as the mean squared error
    loss_pde = (pde_u**2).mean() + (pde_v**2).mean() + (pde_h**2).mean()
    return loss_pde, pde_u.mean(), pde_v.mean(), pde_h.mean()

#setting
width = 32  # example
height = 64  # example
total_param = 12
dt = 1
train_percentage = 0.7
g = 9.81  # Gravitational constant
n = 0.03 #manning roughness

data = np.load("../huz_evolution.npz")

# Get the raw data
raw_data = data["data"]
raw_data = raw_data.reshape((-1, width, height, total_param))

raw_data = raw_data[34:]

data_extract = raw_data[:, :, :, [0, 1, 2, 3, 4]]

#load topography
with open('../topography.txt', "r") as file:
    data_topo = np.array([
        list(map(float, line.split()))
        for line in file if not line.startswith("#")
    ])

#print(data_topo, len(data_topo))
x_topo = data_topo[:,0]
y_topo = data_topo[:,1]
z_topo = data_topo[:,2]

# Reshape z into a 2D grid based on unique x and y
x_unique = np.unique(x_topo)
y_unique = np.unique(y_topo)
z_grid = z_topo.reshape(len(x_unique), len(y_unique))  # Ensure the grid is correctly formed

dx = x_unique[1] - x_unique[0]
dy = y_unique[1] - y_unique[0]

dz_dx = np.gradient(z_grid, axis=1) / dx  # Partial derivative with respect to x
dz_dy = np.gradient(z_grid, axis=0) / dy  # Partial derivative with respect to y

dz_dx = np.expand_dims(np.expand_dims(dz_dx, axis=0), axis=-1)  # Shape becomes (1, n, m, 1)
dz_dx = np.repeat(dz_dx, len(data_extract), axis=0)  # Shape becomes (t, n, m, 1)

dz_dy = np.expand_dims(np.expand_dims(dz_dy, axis=0), axis=-1)  # Shape becomes (1, n, m, 1)
dz_dy = np.repeat(dz_dy, len(data_extract), axis=0)  # Shape becomes (t, n, m, 1)

data_extract = np.concatenate((data_extract, dz_dx), axis=-1)
data_extract = np.concatenate((data_extract, dz_dy), axis=-1)

#make time dataset
start_time = 34
train_time_data = np.arange(start_time, start_time+len(data_extract))

train_time_data4d = train_time_data[:, np.newaxis, np.newaxis, np.newaxis] * np.ones_like(dz_dx)

data_extract = np.concatenate((data_extract, train_time_data4d), axis=-1)

#########
#try out#
#########

raw_data_torch = torch.from_numpy(data_extract)
loss_pde, pde_u, pde_v, pde_h = physics_loss2(raw_data_torch, g, n)
#print('cek shape', loss_pde.size(), pde_u.size(), pde_v.size(), pde_h.size())
print(f"loss PDE total: {loss_pde}, PDE u: {pde_u}, PDE v: {pde_v}, PDE h: {pde_h}")