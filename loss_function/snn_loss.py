import torch
import torch.nn as nn
import torch.nn.functional as F
from conf import conf


class SNNLoss(nn.Module):
    """
    Soft Nearest Neighbor (SNN) Loss implementation as described in the CalibSNN paper.
    
    This loss encourages intra-class compactness and inter-class separation by
    computing similarities based on exponential of negative squared distances.
    """
    
    def __init__(self, tau=conf.get('--tau', 1.75)):
        """
        Initialize SNN Loss.
        
        Args:
            tau (float): Temperature parameter controlling the concentration of the distribution.
                        Smaller tau focuses on very close neighbors, larger tau considers broader set.
        """
        super(SNNLoss, self).__init__()
        self.tau = tau
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, embeddings, labels):
        """
        Compute SNN loss for a batch of embeddings.
        
        Args:
            embeddings (torch.Tensor): Feature embeddings of shape (batch_size, embedding_dim)
            labels (torch.Tensor): Class labels of shape (batch_size,)
            
        Returns:
            torch.Tensor: Scalar loss value
        """
        batch_size = embeddings.size(0)
        
        # Handle edge cases
        if batch_size <= 1:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Normalize embeddings for stability
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise squared distances
        # ||z_i - z_j||^2 = ||z_i||^2 + ||z_j||^2 - 2<z_i, z_j>
        # Since embeddings are normalized, ||z_i||^2 = 1
        dot_product = torch.mm(embeddings, embeddings.t())
        distances = 2 - 2 * dot_product
        
        # Clamp distances to avoid numerical issues
        distances = torch.clamp(distances, min=0.0, max=4.0)
        
        # Create mask for positive and negative pairs
        labels = labels.view(-1, 1)
        device = embeddings.device  # Use embeddings device instead of self.device
        mask_pos = (labels == labels.t()).float() - torch.eye(batch_size, device=device)
        mask_neg = (labels != labels.t()).float()
        
        # Check if there are any positive or negative pairs
        num_pos_pairs = mask_pos.sum()
        num_neg_pairs = mask_neg.sum()
        
        if num_pos_pairs == 0 or num_neg_pairs == 0:
            # If no positive or negative pairs, return small loss
            return torch.tensor(0.1, device=embeddings.device, requires_grad=True)
        
        # Compute similarities using exponential of negative squared distances
        # Use stable computation to avoid overflow/underflow
        scaled_distances = distances / (self.tau ** 2)
        
        # For numerical stability, subtract the maximum before exponentiating
        max_dist = scaled_distances.max()
        similarities = torch.exp(-(scaled_distances - max_dist))
        
        # Mask out self-similarities
        similarities = similarities * (1 - torch.eye(batch_size, device=device))
        
        # Compute SNN loss for each sample
        loss = 0
        valid_samples = 0
        
        for i in range(batch_size):
            # Get positive and negative similarities for sample i
            pos_sim = similarities[i] * mask_pos[i]
            neg_sim = similarities[i] * mask_neg[i]
            
            # Sum of positive similarities
            pos_sum = pos_sim.sum()
            
            # Sum of all similarities (positive + negative)
            all_sum = pos_sim.sum() + neg_sim.sum()
            
            # Only compute loss if there are both positive and negative samples
            if pos_sum > 0 and all_sum > pos_sum:
                # SNN loss for sample i: -log(sum_pos / sum_all)
                # Add small epsilon for numerical stability
                loss_i = -torch.log(pos_sum / (all_sum + 1e-8) + 1e-8)
                loss += loss_i
                valid_samples += 1
        
        # Average loss over valid samples
        if valid_samples > 0:
            return loss / valid_samples
        else:
            return torch.tensor(0.1, device=embeddings.device, requires_grad=True)


class CalibSNNLoss(nn.Module):
    """
    Combined loss for CalibSNN method: Cross-entropy + SNN loss
    """
    
    def __init__(self, tau=conf.get('--tau', 1.75), lambda_snn=conf.get('lambda_snn', 1.0)):
        """
        Initialize CalibSNN combined loss.
        
        Args:
            tau (float): Temperature parameter for SNN loss
            lambda_snn (float): Weight for SNN loss component
        """
        super(CalibSNNLoss, self).__init__()
        self.snn_loss = SNNLoss(tau)
        self.ce_loss = nn.CrossEntropyLoss()
        self.lambda_snn = lambda_snn
    
    def forward(self, embeddings, logits, labels):
        """
        Compute combined loss.
        
        Args:
            embeddings (torch.Tensor): Feature embeddings
            logits (torch.Tensor): Classification logits
            labels (torch.Tensor): Ground truth labels
            
        Returns:
            torch.Tensor: Combined loss value
        """
        # Handle edge cases
        if embeddings.size(0) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Cross-entropy loss for classification
        ce_loss = self.ce_loss(logits, labels)
        
        # SNN loss for feature learning
        snn_loss = self.snn_loss(embeddings, labels)
        
        # Combined loss with clamping to avoid extreme values
        total_loss = ce_loss + self.lambda_snn * snn_loss
        
        # Clamp to avoid numerical issues
        total_loss = torch.clamp(total_loss, min=0.0, max=100.0)
        
        return total_loss 