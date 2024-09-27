import torch
from conf import conf
from collections import Counter
import torch.nn.functional as F
class FedLCalibratedLoss():
    def __init__(self, tau=conf['--tau']):
        """
        Initializes the FedLCalibratedLoss.

        Args:
            tau (float): The tau parameter for calibration.
        """
        self.tau = tau
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Assuming `conf` is a dictionary containing configuration parameters
        self.num_classes = conf.get('num_classes', 0)  # Get the number of classes from the config
        if self.num_classes == 0:
            raise ValueError("Number of classes not found in configuration.")

        self.label_distrib = torch.zeros(self.num_classes, device=self.device)

    def logit_calibrated_loss(self,logit, y):
        """
        Computes the loss for a given batch of logits and labels.

        Args:
            logit (torch.Tensor): The logits predicted by the model.
            y (torch.Tensor): The true labels.

        Returns:
            torch.Tensor: The computed loss.
        """
        label_counter = Counter(y.tolist()) # count number of label occurance in batch
        for cls, count in label_counter.items():
            self.label_distrib[cls] = max(1e-8, count)

        cal_logit = torch.exp(
            logit
            - (
                self.tau
                * torch.pow(self.label_distrib, -1 / 4)
                .unsqueeze(0)
                .expand((logit.shape[0], -1))
            )
        )
        y_logit = torch.gather(cal_logit, dim=-1, index=y.unsqueeze(1))
        loss = -torch.log(y_logit / cal_logit.sum(dim=-1, keepdim=True))
        return loss.sum() / logit.shape[0]

# v2 for faster processing
class FedLCalibratedLoss():
    def __init__(self, tau=conf['--tau']):
        """
        Initializes the FedLCalibratedLoss.

        Args:
            tau (float): The tau parameter for calibration.
        """
        self.tau = tau
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = conf.get('num_classes', 0)  # Get the number of classes from the config
        
        if self.num_classes == 0:
            raise ValueError("Number of classes not found in configuration.")
        
        # Initialize label distribution tensor
        self.label_distrib = torch.zeros(self.num_classes, device=self.device)

    def logit_calibrated_loss(self, logit, y):
        """
        Computes the loss for a given batch of logits and labels.

        Args:
            logit (torch.Tensor): The logits predicted by the model.
            y (torch.Tensor): The true labels.

        Returns:
            torch.Tensor: The computed loss.
        """
        # Calculate label distribution using tensor operations for better performance
        batch_size = y.size(0)
        label_counts = torch.bincount(y, minlength=self.num_classes).float()
        
        # Ensure there are no zero counts for stability
        self.label_distrib = torch.maximum(label_counts, torch.tensor(1e-8, device=self.device))

        # Compute calibration term only once
        calibration_term = torch.pow(self.label_distrib, -1 / 4).unsqueeze(0)


        # Calibrate the logits
        cal_logit = torch.exp(logit - (self.tau * calibration_term))
        
        # Gather logits corresponding to the true labels
        y_logit = torch.gather(cal_logit, dim=-1, index=y.unsqueeze(1))
        
        # Compute the calibrated loss
        loss = -torch.log(y_logit / cal_logit.sum(dim=-1, keepdim=True))
    

        # Return the mean loss over the batch
        return loss.mean()


class FedLCalibratedContrastiveLoss(FedLCalibratedLoss):
    def __init__(self, tau=conf['--tau'], margin=1.0, lambda_=1.0):
        super().__init__(tau)
        self.margin = margin
        self.lambda_ = lambda_

    def contrastive_loss(self, embeddings1, labels1, embeddings2, labels2):
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        distances = torch.norm(embeddings1 - embeddings2, p=2, dim=1)
        label_matches = labels1 == labels2
        pos_loss = distances ** 2 * label_matches.float()
        neg_loss = (self.margin - distances).clamp(min=0) ** 2 * (~label_matches).float()
        loss = pos_loss + neg_loss
        return loss.mean()

    def combined_loss(self, logit1, y1, embeddings1, logit2, y2, embeddings2):
        cal_loss1 = self.logit_calibrated_loss(logit1, y1)
        cal_loss2 = self.logit_calibrated_loss(logit2, y2)
        con_loss = self.contrastive_loss(embeddings1, y1, embeddings2, y2)
        return (cal_loss1 + cal_loss2) / 2 + self.lambda_ * con_loss

#v2 to speed up

class FedLCalibratedContrastiveLoss(FedLCalibratedLoss):
    def __init__(self, tau=conf['--tau'], margin=1.0, lambda_=1.0):
        super().__init__(tau)
        self.margin = margin
        self.lambda_ = lambda_

    def contrastive_loss(self, embeddings1, labels1, embeddings2, labels2):
        """
        Computes the contrastive loss based on embeddings and labels.
        """
        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)

        # Compute pairwise distances between embeddings
        distances = torch.norm(embeddings1 - embeddings2, p=2, dim=1)

        # Determine positive and negative pairs (match or no match)
        label_matches = (labels1 == labels2).float()

        # Compute positive loss (for matched pairs)
        pos_loss = distances ** 2 * label_matches

        # Compute negative loss (for unmatched pairs)
        neg_loss = ((self.margin - distances).clamp(min=0) ** 2) * (1 - label_matches)

        # Combine losses and return mean
        loss = pos_loss + neg_loss
        return loss.mean()

    def combined_loss(self, logit1, y1, embeddings1, logit2, y2, embeddings2):
        """
        Combines calibrated loss and contrastive loss.
        """
        # Compute calibrated losses for both sets of logits
        cal_loss1 = self.logit_calibrated_loss(logit1, y1)
        cal_loss2 = self.logit_calibrated_loss(logit2, y2)

        # Compute contrastive loss based on embeddings
        con_loss = self.contrastive_loss(embeddings1, y1, embeddings2, y2)

        # Combine losses (calibrated and contrastive) and return
        return (cal_loss1 + cal_loss2) / 2 + self.lambda_ * con_loss