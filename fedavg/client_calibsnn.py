import torch
import numpy as np
import pandas as pd
import time
from fedavg.datasets import get_dataset
from conf import conf
import os
from loss_function.select_loss_fn import selected_loss_function
from loss_function.calibration import FeatureCalibration


class ClientCalibSNN(object):
    def __init__(self, conf, model, train_df, val_df, feature_dim=None):
        """
        CalibSNN Client implementation.
        
        :param conf: configuration
        :param model: model 
        :param train_df: Train Dataset
        :param val_df: Val Dataset
        :param feature_dim: Dimension of feature embeddings
        """
        self.conf = conf
        self.local_model = model
        self.train_df = train_df
        self.train_dataset = get_dataset(conf, self.train_df, False)  # Don't load pairs
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=conf["batch_size"], shuffle=True)

        self.val_df = val_df
        self.val_dataset = get_dataset(conf, self.val_df, False)
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=conf["batch_size"], shuffle=True)
        
        # Initialize feature calibration
        if feature_dim is None:
            # Infer feature dimension from model
            if conf['model_name'] == 'mlp':
                feature_dim = 64
            elif conf['model_name'] == 'cnn':
                feature_dim = 512
            else:
                feature_dim = 128  # Default
        
        self.calibration = FeatureCalibration(conf["num_classes"], feature_dim)
        self.local_mean = None
        self.local_cov = None
        self.local_count = None

    def local_train(self, model, client_id, global_epochs, global_calibration=None):
        """
        Local training with CalibSNN.
        
        :param model: Global model
        :param client_id: Client ID
        :param global_epochs: Current global epoch
        :param global_calibration: Global calibration statistics
        :return: Updated model state dict and training info
        """
        # Copy global model parameters
        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        # Update global calibration if provided
        if global_calibration is not None:
            self.calibration.global_mean = global_calibration['mean']
            self.calibration.global_cov = global_calibration['cov']

        # Optimizer setup
        if self.conf["client_optimizer"] == "SGD":
            optimizer = torch.optim.SGD(
                self.local_model.parameters(), 
                lr=self.conf['lr'], 
                momentum=self.conf['momentum'],
                weight_decay=self.conf["weight_decay"])
        elif self.conf["client_optimizer"] == "Adam":
            optimizer = torch.optim.Adam(
                self.local_model.parameters(), 
                lr=self.conf['lr'], 
                weight_decay=self.conf["weight_decay"])
        else:
            raise ValueError("Please select client_optimizer in conf.py!")

        criterion = selected_loss_function(loss=self.conf['train_loss_criterion'])
        local_training_info = []

        # First, compute local statistics if calibration is enabled
        if self.conf['calibsnn']['enable_calibration']:
            self.compute_local_statistics()

        # Training loop
        for e in range(self.conf["local_epochs"]):
            start_time = time.time()
            total_loss, total_dataset_size = 0, 0
            
            try:
                self.local_model.train()
                
                for batch_id, batch in enumerate(self.train_loader):
                    # Timeout check
                    if time.time() - start_time > 30:
                        raise TimeoutError(f"Timeout: Client {client_id} took too long in epoch {e}.")

                    data, target = batch
                    if torch.cuda.is_available():
                        data = data.cuda()
                        target = target.cuda()

                    # Don't reshape target for CE loss - it expects 1D tensor
                    # Only reshape for BCE loss (criterion 1)
                    if self.conf['data_type'] == 'tabular' and self.conf['train_loss_criterion'] == 1:
                        target = target.float().view(-1, 1)

                    total_dataset_size += data.size()[0]

                    optimizer.zero_grad()
                    
                    # Forward pass to get embeddings and logits
                    embeddings, logits = self.local_model(data)
                    
                    # Apply calibration if enabled
                    if self.conf['calibsnn']['enable_calibration'] and self.local_mean is not None:
                        # For calibration, ensure target is 1D
                        target_1d = target.squeeze() if target.dim() > 1 else target
                        calibrated_embeddings = self.calibration.calibrate_features(
                            embeddings, target_1d, self.local_mean, self.local_cov)
                    else:
                        calibrated_embeddings = embeddings
                    
                    # Compute loss based on criterion type
                    if self.conf['train_loss_criterion'] == 3:  # SNN loss only
                        loss = criterion(calibrated_embeddings, target)
                    elif self.conf['train_loss_criterion'] == 4:  # CalibSNN loss
                        loss = criterion(calibrated_embeddings, logits, target)
                    else:  # Standard CE or BCE loss
                        loss = criterion(logits, target)
                    
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                # NOTE: Synthetic data generation is handled SERVER-SIDE, not here
                # Clients only train on real data and compute local statistics

                # Evaluation
                acc, eval_loss = self.model_eval()
                train_loss = total_loss / total_dataset_size
                
                # Clients only do real data training - synthetic is SERVER-SIDE
                synthetic_used = False
                
                local_training_info.append({
                    'global_epoch': global_epochs,
                    'client_id': client_id,
                    'epoch': e,
                    'train_loss': train_loss,
                    'eval_loss': eval_loss,
                    'eval_acc': acc,
                    'global_acc': None,
                    'global_loss': None,
                    'method': 'CalibSNN_Complete' if synthetic_used else 'CalibSNN_Partial',
                    'synthetic_data_used': synthetic_used,
                    'resampling_enabled': self.conf['calibsnn']['enable_resampling'],
                    'calibration_enabled': self.conf['calibsnn']['enable_calibration']
                })
                
                method_status = "Complete CalibSNN" if synthetic_used else "Partial CalibSNN (no synthetic)"
                print(f"Client {client_id}, Epoch {e} [{method_status}]: train_loss = {train_loss:.4f}, "
                      f"eval_loss = {eval_loss:.4f}, eval_acc = {acc:.2f}%")

            except TimeoutError as te:
                print(te)
                break

        return self.local_model.state_dict(), local_training_info
    
# Synthetic data generation methods removed - handled SERVER-SIDE
    
    def compute_local_statistics(self):
        """
        Compute local feature statistics for calibration.
        """
        self.local_model.eval()
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.train_loader:
                data, target = batch
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()
                
                embeddings, _ = self.local_model(data)
                all_features.append(embeddings)
                all_labels.append(target)
        
        if len(all_features) > 0:
            all_features = torch.cat(all_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            # Compute statistics
            self.local_mean, self.local_cov = self.calibration.compute_local_statistics(
                all_features, all_labels)
            
            # Compute counts per class
            self.local_count = {}
            for c in range(self.conf['num_classes']):
                self.local_count[c] = (all_labels == c).sum().item()
        else:
            # If no data, initialize with defaults
            self.local_mean = {c: torch.zeros(self.calibration.feature_dim).to(self.calibration.device) 
                              for c in range(self.conf['num_classes'])}
            self.local_cov = {c: torch.eye(self.calibration.feature_dim).to(self.calibration.device) 
                             for c in range(self.conf['num_classes'])}
            self.local_count = {c: 0 for c in range(self.conf['num_classes'])}
    
    def get_local_statistics(self):
        """
        Get local statistics for server aggregation.
        """
        if self.local_mean is None:
            self.compute_local_statistics()
        
        # Convert to CPU for transmission
        mean_cpu = {c: self.local_mean[c].cpu() for c in self.local_mean}
        cov_cpu = {c: self.local_cov[c].cpu() for c in self.local_cov}
        
        return mean_cpu, cov_cpu, self.local_count
    
    @torch.no_grad()
    def model_eval(self):
        """
        Evaluation logic for CalibSNN.
        """
        self.local_model.eval()
        total_val_loss = 0.0
        total_correct = 0
        total_samples = 0
        criterion = selected_loss_function(loss=self.conf['eval_loss_criterion'])
        
        for batch_id, batch in enumerate(self.val_loader):
            data, target = batch
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            # Only reshape for BCE loss
            if self.conf['data_type'] == 'tabular' and self.conf['eval_loss_criterion'] == 1:
                target = target.float().view(-1, 1)

            embeddings, logits = self.local_model(data)
            
            # Compute loss
            if self.conf['eval_loss_criterion'] == 3:  # SNN loss only
                loss = criterion(embeddings, target)
            elif self.conf['eval_loss_criterion'] == 4:  # CalibSNN loss
                loss = criterion(embeddings, logits, target)
            else:  # Standard loss
                loss = criterion(logits, target)
            
            total_val_loss += loss.item() * data.size(0)

            # Compute accuracy
            if self.conf['classification_type'] == "multi":
                pred = logits.data.max(1)[1]
                total_correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            elif self.conf['classification_type'] == "binary":
                # For binary classification with 2 outputs (CE loss)
                if logits.size(1) == 2:
                    pred = logits.data.max(1)[1]
                    total_correct += pred.eq(target.data).cpu().sum().item()
                else:
                    # For binary classification with 1 output (BCE loss)
                    pred = (torch.sigmoid(logits) > 0.5).float().squeeze()
                    total_correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            else:
                raise ValueError("Please check type of classification! (multi or binary)")

            total_samples += data.size()[0]

        accuracy = 100.0 * (float(total_correct) / float(total_samples))
        avg_val_loss = total_val_loss / total_samples if total_samples > 0 else 0.0
        return accuracy, avg_val_loss 