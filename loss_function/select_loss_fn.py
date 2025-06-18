import torch
from .snn_loss import SNNLoss, CalibSNNLoss
from conf import conf

# Initialize loss functions
snn_loss = SNNLoss()
calibsnn_loss = CalibSNNLoss()

def selected_loss_function(loss):
    """
    Selects the appropriate loss function based on the configuration.

    Returns:
        torch.nn.Module: The selected loss function.
    Raises:
        ValueError: If the specified loss criterion is not supported.
    """
    try:
        if loss == 1:
            criterion = torch.nn.BCEWithLogitsLoss()
        elif loss == 2:
            criterion = torch.nn.CrossEntropyLoss()
        elif loss == 3:
            # SNN loss only (for feature learning)
            criterion = snn_loss
        elif loss == 4:
            # Combined CalibSNN loss (CE + SNN)
            criterion = calibsnn_loss
        else:
            raise ValueError(f"Unsupported loss criterion: {loss}")
    except KeyError:
        raise ValueError("Loss criterion not found in configuration.")
    
    return criterion