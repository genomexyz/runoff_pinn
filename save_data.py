import numpy as np

def process_evolution_file(input_file, output_file):
    """
    Process a Gnuplot evolution file and save as .npz.

    Parameters:
        input_file (str): Path to the evolution .dat file.
        output_file (str): Path to save the .npz file.
    """
    try:
        with open(input_file, 'r') as file:
            lines = file.readlines()

        data_blocks = []
        current_block = []
        timesteps = []

        for line in lines:
            if line.strip() == "":  # Blank line indicates end of a timestep block
                if current_block:
                    data_blocks.append(np.array(current_block))
                    current_block = []
            elif line.startswith("#"):  # Optional: Capture timestep from comment headers
                if "t=" in line:
                    t = float(line.split("t=")[1].strip())
                    timesteps.append(t)
            else:
                current_block.append([float(value) for value in line.split()])

        # Append the last block if not already added
        if current_block:
            data_blocks.append(np.array(current_block))

        # Convert to numpy array for saving
        data_array = np.stack(data_blocks, axis=0)  # Shape: (time, points, variables)
        timesteps = np.array(timesteps) if timesteps else np.arange(len(data_blocks))

        # Save to .npz
        np.savez(output_file, data=data_array, timesteps=timesteps)
        print(f"Data successfully saved to {output_file}")

    except Exception as e:
        print(f"Error processing the file: {e}")

# Example usage
input_file = "huz_evolution.dat"  # Replace with your evolution file path
output_file = "huz_evolution.npz"
process_evolution_file(input_file, output_file)
