import torch

def load_last_layer_params(file_path, last_layer_name='out'):
    # Load the model from the .pth file
    model_state_dict = torch.load(file_path)

    # Extract the weights and biases for the last layer
    last_layer_weights = model_state_dict[f'{last_layer_name}.weight']
    last_layer_biases = model_state_dict[f'{last_layer_name}.bias']

    return last_layer_weights, last_layer_biases

# Replace 'your_model1.pth' and 'your_model2.pth' with the actual file paths of your models
file_path_model1 = './save_model/model.pth'
file_path_model2 = './save_model/retrained_model.pth'

# Load parameters for the last layer from both models
weights_model1, biases_model1 = load_last_layer_params(file_path_model1)
weights_model2, biases_model2 = load_last_layer_params(file_path_model2)

# Compare the weights and biases
weights_equal = torch.equal(weights_model1, weights_model2)
biases_equal = torch.equal(biases_model1, biases_model2)

# Print the results
print(f"Weights are equal: {weights_equal}")
print(f"Biases are equal: {biases_equal}")
