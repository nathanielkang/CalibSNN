import torch
from .calibrated_loss import FedLCalibratedLoss, FedLCalibratedContrastiveLoss
from conf import conf
flcl = FedLCalibratedLoss()
flcl_contrastive = FedLCalibratedContrastiveLoss()
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
            criterion = flcl.logit_calibrated_loss
        elif loss == 4:
            criterion = flcl_contrastive.combined_loss
        else:
            raise ValueError(f"Unsupported loss criterion: {loss}")
    except KeyError:
        raise ValueError("Loss criterion not found in configuration.")
    
    return criterion