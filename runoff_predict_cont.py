import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta

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

param_data_real = param_data.view(-1, width, height, param_data.size(-1))
param_data_real = param_data_real[-1]
param_data_real = param_data_real.view(-1, param_data_real.size(-1))

#print('cek time', param_data_real[:,-1])
#exit()
#param_data_real = param_data_real.unsqueeze(0)

timestep_ahead = 50
output = []
dt_awal = datetime.now()
for iter_t in range(timestep_ahead):
    prediction = model(param_data_real)
    prediction[:, 0] = torch.clamp(prediction[:, 0], min=1e-6)
    prediction_2d = prediction.view(-1, width, height, prediction.size(-1))
    param_data_real[:, 2:5] = prediction
    param_data_real[:, -1] = param_data_real[0, -1]+1
    #print(param_data_real[0, -1]+iter_t+1)
    output.append(prediction_2d.squeeze().detach().numpy())


output = np.array(output)

to_gnuplot(output, 0.2, 0.2, 'prediction_evolution_discovery_best_cont.dat')

dt_akhir = datetime.now()
duration = (dt_akhir - dt_awal).total_seconds()

print(np.shape(output))
print('cek duration', duration)