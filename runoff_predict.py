import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Define the PINN model (simple fully connected neural network)
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        self.n = nn.Parameter(data=torch.tensor([0.])) #manning coeficient, comment out this if use ordinary model
        self.bed_slope_coef = nn.Parameter(data=torch.tensor([0.])) #manning coeficient, comment out this if use ordinary model
        for i in range(len(layers)-1):
            #self.layers.append(nn.init.xavier_normal_(nn.Linear(layers[i], layers[i+1]), gain=1.0))
            #nn.init.zeros_(self.layers[-1].bias.data)
            linear_layer = nn.Linear(layers[i], layers[i+1])
            nn.init.xavier_normal_(linear_layer.weight, gain=1.0)  # Initialize weights
            nn.init.zeros_(linear_layer.bias.data)  # Initialize biases
            self.layers.append(linear_layer)
            if i < len(layers)-2:
                self.layers.append(nn.Tanh())

    #def forward(self, x, y, t):
    def forward(self, inputs):
        #inputs = torch.cat([x, y, t], dim=1)
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

def physics_loss(model, inputs, g, n):
    #DATA SHAPE
    #timestep x width x height x (x y h u v dzdx dzdy time)
    # Enable gradient computation for PDE terms
    inputs.requires_grad_(True)

    # Forward pass
    outputs = model(inputs)  # [u, v, h]
    h, u, v = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3]

    h = torch.clamp(h, min=1e-6)

    #print('cek u', u)
    #print('cek t', inputs[:, 7])

    u_t = torch.autograd.grad(u.sum(), inputs, create_graph=True)[0][:, 7:8]
    u_x = torch.autograd.grad(u.sum(), inputs, create_graph=True)[0][:, 0:1]
    u_y = torch.autograd.grad(u.sum(), inputs, create_graph=True)[0][:, 1:2]

    v_t = torch.autograd.grad(v.sum(), inputs, create_graph=True)[0][:, 7:8]
    v_x = torch.autograd.grad(v.sum(), inputs, create_graph=True)[0][:, 0:1]
    v_y = torch.autograd.grad(v.sum(), inputs, create_graph=True)[0][:, 1:2]

    h_t = torch.autograd.grad(h.sum(), inputs, create_graph=True)[0][:, 7:8]
    h_x = torch.autograd.grad(h.sum(), inputs, create_graph=True)[0][:, 0:1]
    h_y = torch.autograd.grad(h.sum(), inputs, create_graph=True)[0][:, 1:2]

    S_x = inputs[:, 5]
    S_y = inputs[:, 6]

    S0_x = -S_x
    S0_y = -S_y

    Sf_x = n**2 * u * torch.sqrt(u**2 + v**2) / h**(4/3)
    Sf_y = n**2 * v * torch.sqrt(u**2 + v**2) / h**(4/3)

    # Momentum equation residuals
    pde_u = u_t + u * u_x + v * u_y + g * h_x - g * S0_x + g * Sf_x
    pde_v = v_t + u * v_x + v * v_y + g * h_y - g * S0_y + g * Sf_y

    # Continuity equation residual
    pde_h = h_t + (h * u_x + u * h_x) + (h * v_y + v * h_y)

    # PDE loss (sum of squared residuals)
    loss_pde = (pde_u**2).mean() + (pde_v**2).mean() + (pde_h**2).mean()
    if torch.isnan(loss_pde):
        #print('CEK PDE', pde_u.mean(), pde_v.mean(), pde_h.mean())
        #print(u_t, u_x, u_y)
        print(f"h: {h.min()}, {h.max()}, {h.mean()}")  # Check min, max, and mean
        print(f"u: {u.min()}, {u.max()}, {u.mean()}")
        print(f"v: {v.min()}, {v.max()}, {v.mean()}")
        print(f"S detect nan or inf: {torch.isnan(S0_x).any()}, {torch.isinf(S0_x).any()}, {torch.isnan(S0_y).any()}, {torch.isinf(S0_y).any()}")
        print(f"Sf detect nan or inf: {torch.isnan(Sf_x).any()}, {torch.isnan(Sf_y).any()}")
        print(f"cek min max u v: {torch.min(u)}, {torch.min(v)}, {torch.max(u)}, {torch.max(v)}")
        print(f"cek min max nan h : {torch.min(h)}, {torch.max(h)} {torch.isnan(h).any()}")
        #print(f"detect 0 in h u v {torch.any(h == 0)}, {torch.any(u == 0)}, {torch.any(v == 0)}")
        print(f"deteksi per suku {torch.sqrt(u**2 + v**2)} | {h**(4/3)}")
        print("\n=====================================================================")
        print(h)
        exit()
    return loss_pde

def to_gnuplot(matrix, dx, dy, output_file):
    # Open the output file
    with open(output_file, "w") as f:
        # Iterate through the timesteps
        for t in range(matrix.shape[0]):
            # Iterate through the grid (width x height)
            for x in range(matrix.shape[1]):
                for y in range(matrix.shape[2]):
                    # Extract the parameters
                    #params = matrix[t, x, y]
                    # Write data in "timestep x y param1 param2 param3" format
                    f.write(f"{(x*dx):.6f} {(y*dy):.6f} {matrix[t, x, y, 0]:.6f} {matrix[t, x, y, 1]:.6f} {matrix[t, x, y, 2]:.6f}\n")
                f.write("\n")
            # Add a blank line to separate timesteps (optional for better readability in Gnuplot)
            f.write("\n\n")
    print(f"Data successfully written to {output_file}")

#setting
width = 32  # example
height = 64  # example
total_param = 12
dt = 1
train_percentage = 0.7
g = 9.81  # Gravitational constant
n = 0.03 #manning roughness

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model output is U, V, h (water height)
model = PINN([8, 25, 25, 25, 3]).to(DEVICE)

# Load the checkpoint
checkpoint = torch.load("pinn_model_discovery_best.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])

param_data = torch.load('param_test.pth')
prediction = model(param_data)
prediction[:, 0] = torch.clamp(prediction[:, 0], min=1e-6)

print(param_data.size(), prediction.size())
prediction_2d = prediction.view(-1, width, height, prediction.size(-1))
print('shape prediction 2d', prediction_2d.size())

label_data = torch.load('label_test.pth')
label_data = label_data[:, [2,3,4]]
label_2d = label_data.view(-1, width, height, label_data.size(-1))
print('shape label 2d', label_2d.size())

prediction_2d_numpy = prediction_2d.detach().numpy()
label_2d_numpy = label_2d.numpy()

to_gnuplot(prediction_2d_numpy, 0.2, 0.2, 'prediction_evolution_discovery_best.dat')
to_gnuplot(label_2d_numpy, 0.2, 0.2, 'label_evolution.dat')