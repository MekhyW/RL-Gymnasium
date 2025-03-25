import torch
import os
from agent import QNetwork

INPUT_DIM = 8
OUTPUT_DIM = 4

def create_dummy_model_weights(experiment_name):
    """
    Create a dummy .pt file with random model weights for testing purposes.
    
    Parameters:
        experiment_name (str): Name of the experiment for saving the model
    """
    os.makedirs("logging", exist_ok=True)
    model = QNetwork(INPUT_DIM, OUTPUT_DIM)
    for param in model.parameters():
        if param.dim() >= 2:
            torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.zeros_(param)
    model_path = f"logging/dummy_{experiment_name}-model_weights.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Dummy model weights saved to {model_path}")

if __name__ == "__main__":
    create_dummy_model_weights("dummy")