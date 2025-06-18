import torch
import numpy as np
from typing import Dict, List, Tuple


class FeatureCalibration:
    """
    Feature-level calibration for CalibSNN method.
    
    This class handles the calibration of local feature distributions to match
    the global distribution by adjusting mean and covariance.
    """
    
    def __init__(self, num_classes: int, feature_dim: int):
        """
        Initialize feature calibration.
        
        Args:
            num_classes (int): Number of classes in the dataset
            feature_dim (int): Dimension of feature embeddings
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Global statistics (will be updated by server)
        self.global_mean = {c: torch.zeros(feature_dim).to(self.device) 
                           for c in range(num_classes)}
        self.global_cov = {c: torch.eye(feature_dim).to(self.device) 
                          for c in range(num_classes)}
        
        # Cache for expensive computations
        self.global_cov_chol = {c: None for c in range(num_classes)}
        self.local_cov_chol_cache = {}  # Cache for local Cholesky decompositions
        self.cache_valid = False
        
    def compute_local_statistics(self, features: torch.Tensor, labels: torch.Tensor) -> Tuple[Dict, Dict]:
        """
        Compute local mean and covariance for each class.
        
        Args:
            features (torch.Tensor): Feature embeddings (n_samples, feature_dim)
            labels (torch.Tensor): Class labels (n_samples,)
            
        Returns:
            Tuple[Dict, Dict]: Local means and covariances for each class
        """
        local_mean = {}
        local_cov = {}
        
        for c in range(self.num_classes):
            # Get features for class c
            mask = (labels == c)
            class_features = features[mask]
            
            if class_features.size(0) > 0:
                # Compute mean
                mean = class_features.mean(dim=0)
                local_mean[c] = mean
                
                # Compute covariance
                if class_features.size(0) > 1:
                    centered = class_features - mean.unsqueeze(0)
                    cov = torch.mm(centered.t(), centered) / (class_features.size(0) - 1)
                    local_cov[c] = cov
                else:
                    # If only one sample, use identity matrix
                    local_cov[c] = torch.eye(self.feature_dim).to(self.device)
            else:
                # No samples for this class
                local_mean[c] = torch.zeros(self.feature_dim).to(self.device)
                local_cov[c] = torch.eye(self.feature_dim).to(self.device)
                
        return local_mean, local_cov
    
    def update_global_statistics(self, client_stats: Dict[int, Tuple[Dict, Dict, Dict]]):
        """
        Update global statistics from client statistics.
        
        Args:
            client_stats: Dictionary mapping client_id to (mean, cov, count) for each class
        """
        for c in range(self.num_classes):
            total_count = 0
            weighted_mean = torch.zeros(self.feature_dim).to(self.device)
            
            # First pass: compute global mean
            for client_id, (means, covs, counts) in client_stats.items():
                if c in counts and counts[c] > 0:
                    total_count += counts[c]
                    weighted_mean += counts[c] * means[c].to(self.device)
            
            if total_count > 0:
                self.global_mean[c] = weighted_mean / total_count
                
                # Second pass: compute global covariance
                weighted_cov = torch.zeros(self.feature_dim, self.feature_dim).to(self.device)
                
                for client_id, (means, covs, counts) in client_stats.items():
                    if c in counts and counts[c] > 0:
                        # Within-client covariance
                        weighted_cov += (counts[c] - 1) * covs[c].to(self.device)
                        
                        # Between-client covariance
                        mean_diff = means[c].to(self.device) - self.global_mean[c]
                        weighted_cov += counts[c] * torch.outer(mean_diff, mean_diff)
                
                if total_count > 1:
                    self.global_cov[c] = weighted_cov / (total_count - 1)
                else:
                    self.global_cov[c] = torch.eye(self.feature_dim).to(self.device)
        
        # Invalidate cache when global statistics change
        self.cache_valid = False
        self.global_cov_chol = {c: None for c in range(self.num_classes)}
    
    def _get_cached_cholesky(self, matrix: torch.Tensor, cache_key: str) -> torch.Tensor:
        """
        Get cached Cholesky decomposition or compute and cache it.
        
        Args:
            matrix: Covariance matrix to decompose
            cache_key: Key for caching
            
        Returns:
            Cholesky factor
        """
        target_device = matrix.device
        device_cache_key = f"{cache_key}_device_{target_device}"
        
        if device_cache_key not in self.local_cov_chol_cache:
            try:
                regularized = matrix + 1e-6 * torch.eye(self.feature_dim).to(target_device)
                self.local_cov_chol_cache[device_cache_key] = torch.linalg.cholesky(regularized)
            except:
                # Fallback to identity if numerical issues
                self.local_cov_chol_cache[device_cache_key] = torch.eye(self.feature_dim).to(target_device)
        
        return self.local_cov_chol_cache[device_cache_key]
    
    def calibrate_features(self, features: torch.Tensor, labels: torch.Tensor,
                          local_mean: Dict, local_cov: Dict) -> torch.Tensor:
        """
        Calibrate features to match global distribution.
        
        Args:
            features (torch.Tensor): Original features (n_samples, feature_dim)
            labels (torch.Tensor): Class labels (n_samples,)
            local_mean (Dict): Local means for each class
            local_cov (Dict): Local covariances for each class
            
        Returns:
            torch.Tensor: Calibrated features
        """
        # Ensure all operations happen on the same device as features
        target_device = features.device
        calibrated_features = torch.zeros_like(features)
        
        for c in range(self.num_classes):
            # Create proper 1D boolean mask
            mask = (labels == c).squeeze()
            if mask.sum() == 0:
                continue
                
            # Ensure mask is 1D
            if mask.dim() > 1:
                mask = mask.squeeze()
                
            class_features = features[mask]
            
            # Move local statistics to target device
            local_mean_device = local_mean[c].to(target_device)
            local_cov_device = local_cov[c].to(target_device)
            
            # Center features
            centered = class_features - local_mean_device.unsqueeze(0)
            
            # Whitening transform using cached statistics
            try:
                # Use cached local Cholesky decomposition
                local_cache_key = f"local_{c}_{hash(str(local_cov[c].data.cpu().numpy().tobytes()))}"
                L_local = self._get_cached_cholesky(local_cov_device, local_cache_key)
                whitened = torch.linalg.solve_triangular(L_local, centered.t(), upper=False).t()
                
                # Use cached global Cholesky decomposition
                if self.global_cov_chol[c] is None or self.global_cov_chol[c].device != target_device:
                    regularized_global = self.global_cov[c].to(target_device) + 1e-6 * torch.eye(self.feature_dim).to(target_device)
                    self.global_cov_chol[c] = torch.linalg.cholesky(regularized_global)
                
                L_global = self.global_cov_chol[c].to(target_device)
                transformed = torch.mm(whitened, L_global.t())
                
                # Add global mean (ensure it's on correct device)
                global_mean_device = self.global_mean[c].to(target_device)
                calibrated = transformed + global_mean_device.unsqueeze(0)
                
                calibrated_features[mask] = calibrated
            except Exception as e:
                # If numerical issues, just use mean calibration
                global_mean_device = self.global_mean[c].to(target_device)
                calibrated_features[mask] = centered + global_mean_device.unsqueeze(0)
        
        return calibrated_features
    
    def sample_from_global(self, num_samples_per_class: int, target_device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample synthetic features from the global distribution.
        
        Args:
            num_samples_per_class (int): Number of samples to generate per class
            target_device: Device to place the generated samples on
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Synthetic features and labels
        """
        if target_device is None:
            target_device = self.device
            
        all_features = []
        all_labels = []
        
        for c in range(self.num_classes):
            # Sample from multivariate normal distribution
            try:
                # Move global statistics to target device
                global_cov_device = self.global_cov[c].to(target_device)
                global_mean_device = self.global_mean[c].to(target_device)
                
                L = torch.linalg.cholesky(global_cov_device + 1e-6 * torch.eye(self.feature_dim).to(target_device))
                z = torch.randn(num_samples_per_class, self.feature_dim).to(target_device)
                samples = torch.mm(z, L.t()) + global_mean_device.unsqueeze(0)
                
                all_features.append(samples)
                all_labels.append(torch.full((num_samples_per_class,), c, dtype=torch.long).to(target_device))
            except:
                # If numerical issues, sample from standard normal
                global_mean_device = self.global_mean[c].to(target_device)
                samples = torch.randn(num_samples_per_class, self.feature_dim).to(target_device)
                samples = samples + global_mean_device.unsqueeze(0)
                
                all_features.append(samples)
                all_labels.append(torch.full((num_samples_per_class,), c, dtype=torch.long).to(target_device))
        
        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        # Shuffle
        perm = torch.randperm(features.size(0)).to(target_device)
        return features[perm], labels[perm] 