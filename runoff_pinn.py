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
        self.n = nn.Parameter(data=torch.tensor([0.])) #manning coeficient
        self.bed_slope_coef = nn.Parameter(data=torch.tensor([0.])) #manning coeficient
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

    #Sf_x = n**2 * u * torch.sqrt(u**2 + v**2) / h**(4/3)
    #Sf_y = n**2 * v * torch.sqrt(u**2 + v**2) / h**(4/3)

    Sf_x = model.n**2 * u * torch.sqrt(u**2 + v**2) / h**(4/3)
    Sf_y = model.n**2 * v * torch.sqrt(u**2 + v**2) / h**(4/3)

    # Momentum equation residuals
    #pde_u = u_t + u * u_x + v * u_y #+ g * h_x - g * S0_x + g * Sf_x
    #pde_v = v_t + u * v_x + v * v_y #+ g * h_y - g * S0_y + g * Sf_y

    pde_u = u_t + u * u_x + v * u_y + g * h_x - g * S0_x * model.bed_slope_coef + g * Sf_x
    pde_v = v_t + u * v_x + v * v_y + g * h_y - g * S0_y * model.bed_slope_coef + g * Sf_y

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

def save_model(model, optimizer, epoch, loss, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }, path)



#setting
width = 32  # example
height = 64  # example
total_param = 12
dt = 1
train_percentage = 0.7
g = 9.81  # Gravitational constant
n = 0.03 #manning roughness
#timestep = 100  # example

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load data
# Load the .npz file
data = np.load("huz_evolution.npz")

# Get the raw data
raw_data = data["data"]
raw_data = raw_data.reshape((-1, width, height, total_param))

raw_data = raw_data[34:]

#print(raw_data)
#print(np.shape(raw_data))
#print(raw_data[:, 0, 0, 11])

#data_extract = raw_data[:, :, :, [0, 1, 2, 3, 4, 11]]
data_extract = raw_data[:, :, :, [0, 1, 2, 3, 4]]

#load topography
with open('topography.txt', "r") as file:
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



train_len = int(len(data_extract) * train_percentage)
train_data_extract = data_extract[:train_len]
test_data_extract = data_extract[train_len:]

param_train_data_extract = train_data_extract[:-1]
label_train_data_extract = train_data_extract[1:]

param_test_data_extract = test_data_extract[:-1]
label_test_data_extract = test_data_extract[1:]

param_train_data_extract = np.reshape(param_train_data_extract, (-1, param_train_data_extract.shape[-1]))
label_train_data_extract = np.reshape(label_train_data_extract, (-1, label_train_data_extract.shape[-1]))

param_test_data_extract = np.reshape(param_test_data_extract, (-1, param_test_data_extract.shape[-1]))
label_test_data_extract = np.reshape(label_test_data_extract, (-1, label_test_data_extract.shape[-1]))


param_train_data_extract_torch = torch.from_numpy(param_train_data_extract).float()
label_train_data_extract_torch = torch.from_numpy(label_train_data_extract).float()

param_test_data_extract_torch = torch.from_numpy(param_test_data_extract).float()
label_test_data_extract_torch = torch.from_numpy(label_test_data_extract).float()

torch.save(param_test_data_extract_torch, 'param_test.pth')
torch.save(label_test_data_extract_torch, 'label_test.pth')

# Create TensorDatasets
train_dataset = TensorDataset(param_train_data_extract_torch, label_train_data_extract_torch)
test_dataset = TensorDataset(param_test_data_extract_torch, label_test_data_extract_torch)

# Define DataLoaders
batch_size = 32  # You can adjust this
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(np.shape(train_data_extract), np.shape(test_data_extract))

#model output is U, V, h (water height)
model = PINN([8, 25, 25, 25, 3]).to(DEVICE)

# Loss and optimizer
criterion = nn.MSELoss(reduction='mean')  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.00001)

epochs = 10000
err_limit = 10
for epoch in range(epochs):
    errs = []
    errs_mse = []
    errs_phy = []
    for batch in train_loader:
        batch_param, batch_label = batch
        batch_param = batch_param
        batch_label = batch_label[:, [2,3,4]] #h,u,v

        optimizer.zero_grad()

        prediction = model(batch_param)

        loss = criterion(prediction, batch_label)
        phy_loss = physics_loss(model, batch_param, g, n)
        total_loss = loss + phy_loss
        total_loss.backward()
        optimizer.step()

        errs.append(total_loss.item())
        errs_mse.append(loss.item())
        errs_phy.append(phy_loss.item())
    errs_mean = np.mean(np.array(errs))
    errs_mse_mean = np.mean(np.array(errs_mse))
    errs_phy_mean = np.mean(np.array(errs_phy))
    print(f"Epoch {epoch}, Loss: {errs_mean}, MSE: {errs_mse_mean}, PDE: {errs_phy_mean}")
    if errs_mean < err_limit:
        save_model(model, optimizer, epoch, loss.item(), 'pinn_model_discovery_best.pth')
        err_limit = errs_mean
    #if errs_mean < 1:
    #    break
    #break
    
# Save model and additional information
model_save_path = "pinn_model_discovery.pth"

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss.item()
}, model_save_path)
print(f"Model and additional information saved to {model_save_path}")