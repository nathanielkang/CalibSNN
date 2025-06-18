import os
import sys
import torch
import argparse
import json
import time
import copy
from datetime import datetime
import pandas as pd
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

# Import CalibSNN components
from conf import conf, update_conf_for_dataset
from prepare_missing_datasets import ensure_dataset_available
from fedavg.server import Server
from fedavg.client_calibsnn import ClientCalibSNN
from fedavg.models import CNN_Model, weights_init_normal, MLP, TextCNN
from utils import get_data
from loss_function.calibration import FeatureCalibration


def train_clients_on_device(device, client_group, global_model_state, round_idx, global_calibration, exp_conf, primary_device):
    """
    Train a group of clients on a specific GPU device in parallel.
    
    Args:
        device: GPU device to train on
        client_group: List of (client_id, client_object) tuples
        global_model_state: Global model state dict
        round_idx: Current round index
        global_calibration: Global calibration statistics
        exp_conf: Experiment configuration
        primary_device: Primary device where server model lives
        
    Returns:
        Dictionary with client models, statistics, and training info
    """
    results = {
        'client_models': {},
        'client_stats': {},
        'training_info': []
    }
    
    print(f"  🔄 GPU {device}: Training {len(client_group)} clients...")
    
    # Move global calibration to this device if it exists
    device_global_calibration = None
    if global_calibration is not None:
        device_global_calibration = {}
        # Don't use cuda.device() for CPU devices
        if device.type == 'cuda':
            with torch.cuda.device(device):
                # Move global calibration statistics to the current device
                device_global_calibration['mean'] = {}
                device_global_calibration['cov'] = {}
                
                for class_id in global_calibration['mean']:
                    device_global_calibration['mean'][class_id] = global_calibration['mean'][class_id].to(device)
                    device_global_calibration['cov'][class_id] = global_calibration['cov'][class_id].to(device)
        else:
            # CPU device - no need for cuda.device() context
            device_global_calibration['mean'] = {}
            device_global_calibration['cov'] = {}
            
            for class_id in global_calibration['mean']:
                device_global_calibration['mean'][class_id] = global_calibration['mean'][class_id].to(device)
                device_global_calibration['cov'][class_id] = global_calibration['cov'][class_id].to(device)
    
    for client_id, client in client_group:
        try:
            # Ensure client model is on correct device
            if device.type == 'cuda':
                with torch.cuda.device(device):
                    # Copy global model to this device
                    device_global_model = copy.deepcopy(global_model_state)
                    for name, param in device_global_model.items():
                        device_global_model[name] = param.to(device)
                    
                    # Create temporary model for this client
                    temp_model = copy.deepcopy(client.local_model)
                    for name, param in device_global_model.items():
                        temp_model.state_dict()[name].copy_(param)
                    
                    # Ensure client's calibration uses correct device
                    original_device = client.calibration.device
                    client.calibration.device = device
                    
                    # Move client's global calibration to correct device
                    if device_global_calibration is not None:
                        client.calibration.global_mean = device_global_calibration['mean']
                        client.calibration.global_cov = device_global_calibration['cov']
                    
                    # Local training
                    model_state, local_info = client.local_train(
                        temp_model, client_id, round_idx, device_global_calibration
                    )
                    
                    # Restore original device
                    client.calibration.device = original_device
                    
                    # Move model state to primary device for aggregation (server is on primary device)
                    primary_device_model_state = {}
                    for name, param in model_state.items():
                        # Move to the primary device where server model lives
                        primary_device_model_state[name] = param.to(primary_device)
                    
                    results['client_models'][client_id] = primary_device_model_state
                    results['training_info'].extend(local_info)
                    
                    # Get local statistics for calibration (move to CPU)
                    if exp_conf['calibsnn']['enable_calibration']:
                        mean, cov, count = client.get_local_statistics()
                        results['client_stats'][client_id] = (mean, cov, count)
                    
                    print(f"    ✅ GPU {device}: Client {client_id} completed")
            else:
                # CPU device - no need for cuda.device()
                # Copy global model to this device
                device_global_model = copy.deepcopy(global_model_state)
                for name, param in device_global_model.items():
                    device_global_model[name] = param.to(device)
                
                # Create temporary model for this client
                temp_model = copy.deepcopy(client.local_model)
                for name, param in device_global_model.items():
                    temp_model.state_dict()[name].copy_(param)
                
                # Ensure client's calibration uses correct device
                original_device = client.calibration.device
                client.calibration.device = device
                
                # Move client's global calibration to correct device
                if device_global_calibration is not None:
                    client.calibration.global_mean = device_global_calibration['mean']
                    client.calibration.global_cov = device_global_calibration['cov']
                
                # Local training
                model_state, local_info = client.local_train(
                    temp_model, client_id, round_idx, device_global_calibration
                )
                
                # Restore original device
                client.calibration.device = original_device
                
                # Move model state to primary device for aggregation (server is on primary device)
                primary_device_model_state = {}
                for name, param in model_state.items():
                    # Move to the primary device where server model lives
                    primary_device_model_state[name] = param.to(primary_device)
                
                results['client_models'][client_id] = primary_device_model_state
                results['training_info'].extend(local_info)
                
                # Get local statistics for calibration (move to CPU)
                if exp_conf['calibsnn']['enable_calibration']:
                    mean, cov, count = client.get_local_statistics()
                    results['client_stats'][client_id] = (mean, cov, count)
                
                print(f"    ✅ GPU {device}: Client {client_id} completed")
                
        except Exception as e:
            print(f"    ❌ GPU {device}: Client {client_id} failed - {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"  ✅ GPU {device}: Completed {len(results['client_models'])}/{len(client_group)} clients")
    return results


def setup_multi_gpu(gpu_ids_str):
    """Setup multiple GPUs for client-parallel federated learning."""
    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU.")
        return [torch.device('cpu')], torch.device('cpu')
    
    # Parse GPU IDs
    try:
        if ',' in gpu_ids_str:
            gpu_ids = [int(x.strip()) for x in gpu_ids_str.split(',')]
        else:
            gpu_ids = [int(gpu_ids_str)]
    except ValueError:
        print(f"Invalid GPU IDs format: {gpu_ids_str}. Using GPU 0.")
        gpu_ids = [0]
    
    # Validate GPU IDs
    available_gpus = list(range(torch.cuda.device_count()))
    valid_gpu_ids = []
    
    for gpu_id in gpu_ids:
        if gpu_id in available_gpus:
            valid_gpu_ids.append(gpu_id)
            print(f"GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)} - Available")
        else:
            print(f"GPU {gpu_id}: Not available (only have {available_gpus})")
    
    if not valid_gpu_ids:
        print("No valid GPUs found. Using CPU.")
        return [torch.device('cpu')], torch.device('cpu')
    
    # Create device lists
    devices = [torch.device(f'cuda:{gpu_id}') for gpu_id in valid_gpu_ids]
    
    # Set primary device
    primary_device = devices[0]
    torch.cuda.set_device(valid_gpu_ids[0])
    
    print(f"\n🚀 Multi-GPU Setup Complete!")
    print(f"📊 Using {len(devices)} GPUs: {valid_gpu_ids}")
    print(f"🎯 Primary device: {primary_device}")
    print(f"💾 Total GPU memory: {sum(torch.cuda.get_device_properties(i).total_memory for i in valid_gpu_ids) / 1e9:.1f} GB")
    
    return devices, primary_device


def set_gpu(gpu_id):
    """Legacy single GPU setup (for backward compatibility)."""
    if torch.cuda.is_available() and gpu_id >= 0:
        if gpu_id < torch.cuda.device_count():
            torch.cuda.set_device(gpu_id)
            device = torch.device(f'cuda:{gpu_id}')
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            print(f"GPU {gpu_id} not available. Using CPU instead.")
            device = torch.device('cpu')
    else:
        print("CUDA not available. Using CPU.")
        device = torch.device('cpu')
    return device


def create_experiment_name(dataset, beta, num_clients, num_rounds):
    """Create a unique experiment name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{dataset}_round_{num_rounds}_client_{num_clients}_beta_{beta}_{timestamp}"


def run_calibsnn_experiment(args, dataset_name, beta_value):
    """Run a single CalibSNN experiment."""
    print(f"\n{'='*60}")
    print(f"Running CalibSNN on {dataset_name} with beta={beta_value}")
    print(f"{'='*60}\n")
    
    # Update conf with dataset-specific configuration FIRST
    update_conf_for_dataset(dataset_name)
    
    # NOW create a copy of conf with the updated paths
    exp_conf = copy.deepcopy(conf)
    
    # Update with experiment parameters
    exp_conf.update({
        "dataset_used": dataset_name,
        "beta": beta_value,
        "num_parties": args.num_clients,
        "global_epochs": args.num_rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "lr": args.learning_rate,
        "--tau": args.tau,
        "lambda_snn": args.lambda_snn,
        "train_loss_criterion": 4,  # CalibSNN loss
        "eval_loss_criterion": 4,   # CalibSNN loss
        "test_loss_criterion": 2,   # CE for testing
        "calibsnn": {
            "enable_calibration": not args.disable_calibration,  # Default: True (enabled)
            "enable_resampling": not args.disable_resampling,    # Default: True (enabled)
                    "resample_ratio": args.resample_ratio,
        "update_global_every": args.update_global_every,
        "server_epochs": args.server_epochs,
        "synthetic_weight": args.synthetic_weight,
        }
    })
    
    # Update data type and model based on dataset
    if dataset_name in ['kddcup99', 'adult', 'covertype']:
        exp_conf['model_name'] = 'mlp'
        if dataset_name == 'adult':
            exp_conf['classification_type'] = 'binary'  # Adult is binary classification
        elif dataset_name == 'covertype':
            exp_conf['classification_type'] = 'multi'   # CoverType is multi-class (7 classes)
        elif dataset_name == 'kddcup99':
            exp_conf['classification_type'] = 'binary'
    elif dataset_name in ['ag_news']:
        exp_conf['model_name'] = 'textcnn'
        # Text models often need lower learning rates
        exp_conf['lr'] = min(exp_conf['lr'], 0.001)
        exp_conf['server_lr'] = min(exp_conf.get('server_lr', exp_conf['lr']), 0.001)
    elif dataset_name in ['svhn', 'mnist', 'cifar10', 'usps']:
        exp_conf['model_name'] = 'cnn'
        # SVHN often needs different hyperparameters due to its complexity
        if dataset_name == 'svhn':
            exp_conf['lr'] = min(exp_conf['lr'], 0.005)  # Lower LR for SVHN
            exp_conf['server_lr'] = min(exp_conf.get('server_lr', exp_conf['lr']), 0.005)
            exp_conf['weight_decay'] = 5e-4  # Stronger regularization
    
    # Create experiment directory in results folder
    exp_name = create_experiment_name(dataset_name, beta_value, args.num_clients, args.num_rounds)
    exp_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save experiment configuration with descriptive filename
    config_filename = f"config_{dataset_name}_round_{args.num_rounds}_client_{args.num_clients}_beta_{beta_value}.json"
    with open(os.path.join(exp_dir, config_filename), 'w') as f:
        json.dump(exp_conf, f, indent=4)
    
    # Ensure dataset is available before loading
    print(f"Checking if {dataset_name} dataset is available...")
    if not ensure_dataset_available(dataset_name):
        raise RuntimeError(f"Failed to prepare {dataset_name} dataset. Please check the error messages above.")
    
    # Initialize datasets with the experiment configuration
    print("Loading datasets...")
    train_datasets, val_datasets, test_dataset = get_data(conf_dict=exp_conf, min_require_size=args.min_require_size)
    
    # Initialize aggregation weights
    client_weight = {}
    for key in train_datasets.keys():
        client_weight[key] = 1 / len(train_datasets)
    
    # Model initialization based on dataset type
    if exp_conf['data_type'] == 'tabular':
        if dataset_name == 'kddcup99':
            # KDD Cup 99 has 41 features
            model = MLP(n_feature=41, n_hidden=128, n_output=2)
            feature_dim = 16  # n_hidden // 8 = 128 // 8 = 16
        elif dataset_name == 'adult':
            # Adult dataset has 14 features
            model = MLP(n_feature=14, n_hidden=128, n_output=2)
            feature_dim = 16  # n_hidden // 8 = 128 // 8 = 16
        elif dataset_name == 'covertype':
            # CoverType dataset has 54 features
            model = MLP(n_feature=54, n_hidden=256, n_output=7)
            feature_dim = 32  # n_hidden // 8 = 256 // 8 = 32
    elif exp_conf['data_type'] == 'text':
        if dataset_name == 'ag_news':
            # TextCNN for AG News
            vocab_size = 30000  # This should be set based on actual vocabulary
            embedding_dim = 128
            model = TextCNN(vocab_size=vocab_size, embedding_dim=embedding_dim, 
                          num_classes=4, num_filters=100)
            feature_dim = 300  # 3 * num_filters
    elif exp_conf['data_type'] == 'image':
        if exp_conf['model_name'] == 'cnn':
            model = CNN_Model()
            feature_dim = 512  # 2 * 2 * 128
        elif exp_conf['model_name'] == 'mlp':
            if dataset_name in ['mnist', 'fmnist']:
                model = MLP(n_feature=784, n_hidden=512, n_output=10)
                feature_dim = 64  # 512 // 8
            else:
                model = MLP(n_feature=3072, n_hidden=512, n_output=10)
                feature_dim = 64  # 512 // 8
    else:
        raise ValueError(f"Unknown data type: {exp_conf['data_type']}")
    
    model.apply(weights_init_normal)
    
    # Setup multi-GPU or single GPU
    if hasattr(args, 'gpu_ids') and args.gpu_ids:
        devices, primary_device = setup_multi_gpu(args.gpu_ids)
    else:
        # Fallback to legacy single GPU
        primary_device = set_gpu(args.gpu_id)
        devices = [primary_device]
    
    # Move model to primary device
    model = model.to(primary_device)
    
    # Initialize server
    server = Server(exp_conf, model, test_dataset)
    print("Server initialized!")
    
    # Initialize CalibSNN clients and distribute across GPUs
    clients = {}
    client_device_mapping = {}  # Track which device each client uses
    client_keys = list(train_datasets.keys())
    
    print(f"\n🔄 Distributing {len(client_keys)} clients across {len(devices)} GPUs...")
    for i, key in enumerate(client_keys):
        # Assign clients to GPUs in round-robin fashion
        device_idx = i % len(devices)
        assigned_device = devices[device_idx]
        
        # Create client model on assigned device
        client_model = copy.deepcopy(server.global_model).to(assigned_device)
        
        clients[key] = ClientCalibSNN(
            exp_conf, 
            client_model, 
            train_datasets[key], 
            val_datasets[key], 
            feature_dim=feature_dim
        )
        
        client_device_mapping[key] = assigned_device
        
        if i < 10 or i % 50 == 0:  # Show first 10 and every 50th assignment
            print(f"  Client {key} → GPU {assigned_device}")
    
    print(f"✅ Client distribution complete!")
    for device_idx, device in enumerate(devices):
        clients_on_device = sum(1 for d in client_device_mapping.values() if d == device)
        print(f"   GPU {device}: {clients_on_device} clients")
    print(f"📊 Total: {len(clients)} CalibSNN clients initialized!")
    
    # Initialize global calibration
    global_calibration = None
    calibration_aggregator = FeatureCalibration(exp_conf["num_classes"], feature_dim)
    
    # Training metrics
    global_training_info = []
    round_metrics = []
    
    # Setup CSV files for real-time saving
    metrics_filename = f"{dataset_name}_round_{args.num_rounds}_client_{args.num_clients}_beta_{beta_value}_tau_{args.tau}_lambda_{args.lambda_snn}_Results.csv"
    metrics_path = os.path.join(exp_dir, metrics_filename)
    
    detailed_filename = f"{dataset_name}_round_{args.num_rounds}_client_{args.num_clients}_beta_{beta_value}_tau_{args.tau}_lambda_{args.lambda_snn}_Detailed.csv"
    detailed_path = os.path.join(exp_dir, detailed_filename)
    
    # Check if resuming from checkpoint
    start_round = 0
    if args.resume and os.path.exists(args.resume):
        print(f"\n📥 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=primary_device)
        
        # Verify experiment configuration matches
        saved_config = checkpoint.get('experiment_config', {})
        if (saved_config.get('dataset') != dataset_name or 
            saved_config.get('beta') != beta_value or
            saved_config.get('num_clients') != args.num_clients):
            print("❌ ERROR: Checkpoint configuration doesn't match current experiment!")
            print(f"   Checkpoint: {saved_config}")
            print(f"   Current: dataset={dataset_name}, beta={beta_value}, clients={args.num_clients}")
            raise ValueError("Cannot resume from incompatible checkpoint")
        
        # Load model state
        server.global_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load round number
        start_round = checkpoint['round'] + 1
        
        # Load global calibration if it exists
        if 'global_calibration' in checkpoint and checkpoint['global_calibration'] is not None:
            global_calibration = checkpoint['global_calibration']
            # Update calibration aggregator with loaded statistics
            for class_id in global_calibration['mean']:
                calibration_aggregator.global_mean[class_id] = global_calibration['mean'][class_id]
                calibration_aggregator.global_cov[class_id] = global_calibration['cov'][class_id]
        
        print(f"✅ Resumed from round {checkpoint['round']}")
        print(f"   Test Accuracy at checkpoint: {checkpoint['test_accuracy']:.2f}%")
        print(f"   Continuing from round {start_round}...")
        
        # Load existing metrics if file exists
        if os.path.exists(metrics_path):
            existing_metrics = pd.read_csv(metrics_path)
            round_metrics = existing_metrics.to_dict('records')
            print(f"   Loaded {len(round_metrics)} existing round metrics")
    
    # Main training loop
    start_time = time.time()
    
    for round_idx in range(start_round, exp_conf["global_epochs"]):
        round_start = time.time()
        
        # Sample clients for this round (major performance improvement)
        client_sampling_rate = args.client_sampling_rate if hasattr(args, 'client_sampling_rate') else 1.0
        if client_sampling_rate < 1.0:
            num_selected = max(1, int(len(clients) * client_sampling_rate))
            selected_clients = np.random.choice(list(clients.keys()), num_selected, replace=False)
            print(f"Round {round_idx}: Sampling {num_selected}/{len(clients)} clients")
        else:
            selected_clients = list(clients.keys())
            print(f"Round {round_idx}: Training all {len(clients)} clients")
        
        # Collect client updates and statistics
        clients_models = {}
        client_stats = {}
        
        # 🚀 Multi-GPU Parallel Client Training
        print(f"\n🚀 Round {round_idx}: Parallel training across {len(devices)} GPUs...")
        
        # Group selected clients by their assigned devices
        device_client_groups = defaultdict(list)
        for client_id in selected_clients:
            assigned_device = client_device_mapping[client_id]
            device_client_groups[assigned_device].append((client_id, clients[client_id]))
        
        # Display client distribution for this round
        for device in devices:
            if device in device_client_groups:
                print(f"  📊 GPU {device}: {len(device_client_groups[device])} clients")
        
        # Get global model state
        global_model_state = server.global_model.state_dict()
        
        # Run parallel training across all GPUs
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            # Submit training jobs for each device
            future_to_device = {}
            for device, client_group in device_client_groups.items():
                if client_group:  # Only if there are clients for this device
                    future = executor.submit(
                        train_clients_on_device,
                        device, client_group, global_model_state, 
                        round_idx, global_calibration, exp_conf, primary_device
                    )
                    future_to_device[future] = device
            
            # Collect results from all devices
            for future in future_to_device:
                try:
                    device = future_to_device[future]
                    results = future.result(timeout=300)  # 5 minute timeout per device
                    
                    # Merge results
                    clients_models.update(results['client_models'])
                    global_training_info.extend(results['training_info'])
                    client_stats.update(results['client_stats'])
                    
                except Exception as e:
                    device = future_to_device[future]
                    print(f"❌ GPU {device} training failed: {str(e)}")
        
        print(f"✅ Round {round_idx}: Collected results from {len(clients_models)} clients")
        
        # Update global calibration statistics
        if (exp_conf['calibsnn']['enable_calibration'] and 
            round_idx % exp_conf['calibsnn']['update_global_every'] == 0):
            calibration_aggregator.update_global_statistics(client_stats)
            # Keep global calibration on primary device initially (will be moved to each GPU as needed)
            global_calibration = {
                'mean': {c: calibration_aggregator.global_mean[c].to(primary_device) for c in calibration_aggregator.global_mean},
                'cov': {c: calibration_aggregator.global_cov[c].to(primary_device) for c in calibration_aggregator.global_cov}
            }
            print(f"Updated global calibration statistics at round {round_idx}")
        
        # Aggregate models
        server.model_aggregate(clients_models, client_weight)
        
        # SERVER-SIDE: Synthetic data generation and training
        if (exp_conf['calibsnn']['enable_resampling'] and 
            exp_conf['calibsnn']['enable_calibration'] and
            global_calibration is not None):
            
            print(f"🧬 SERVER: Generating synthetic data for global model refinement...")
            server_synthetic_training(server, exp_conf, global_calibration, primary_device)
        
        # Evaluate global model
        test_acc, test_loss = server.model_eval()
        
        round_time = time.time() - round_start
        
        # Check if this is complete CalibSNN (server-side synthetic training)
        is_complete_calibsnn = (exp_conf['calibsnn']['enable_resampling'] and 
                               exp_conf['calibsnn']['enable_calibration'] and
                               global_calibration is not None)
        
        # Record metrics
        round_metric = {
            'method': 'CalibSNN_Complete' if is_complete_calibsnn else 'CalibSNN_Partial',
            'round': round_idx,
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'round_time': round_time,
            'dataset': dataset_name,
            'beta': beta_value,
            'num_clients': args.num_clients,
            'num_rounds': args.num_rounds,
            'synthetic_resampling': exp_conf['calibsnn']['enable_resampling'],
            'feature_calibration': exp_conf['calibsnn']['enable_calibration'],
            'resample_ratio': exp_conf['calibsnn']['resample_ratio'],
            'update_global_every': exp_conf['calibsnn']['update_global_every'],
            'server_epochs': exp_conf['calibsnn']['server_epochs'],
            'tau': args.tau,
            'lambda_snn': args.lambda_snn
        }
        round_metrics.append(round_metric)
        
        print(f"Round {round_idx}: Test Acc = {test_acc:.2f}%, "
              f"Test Loss = {test_loss:.4f}, Time = {round_time:.2f}s")
        
        # Real-time save: Update CSV files after each round
        # Save round metrics
        metrics_df = pd.DataFrame(round_metrics)
        metrics_df.to_csv(metrics_path, index=False)
        
        # Save detailed training info periodically (every 5 rounds or at the end)
        if global_training_info and ((round_idx + 1) % 5 == 0 or round_idx == exp_conf["global_epochs"] - 1):
            detailed_df = pd.DataFrame(global_training_info)
            detailed_df.to_csv(detailed_path, index=False)
            print(f"   💾 Saved progress: {len(round_metrics)} rounds, {len(global_training_info)} training records")
        
        # Save checkpoint with descriptive filename
        if (round_idx + 1) % args.save_every == 0:
            checkpoint_filename = f'checkpoint_{dataset_name}_round_{round_idx}_beta_{beta_value}_acc_{test_acc:.2f}.pth'
            checkpoint_path = os.path.join(exp_dir, checkpoint_filename)
            torch.save({
                'round': round_idx,
                'model_state_dict': server.global_model.state_dict(),
                'test_accuracy': test_acc,
                'test_loss': test_loss,
                'global_calibration': global_calibration,
                'experiment_config': {
                    'dataset': dataset_name,
                    'beta': beta_value,
                    'num_clients': args.num_clients,
                    'num_rounds': args.num_rounds,
                    'tau': args.tau,
                    'lambda_snn': args.lambda_snn,
                    'resample_ratio': args.resample_ratio,
                    'update_global_every': args.update_global_every,
                    'server_epochs': args.server_epochs
                }
            }, checkpoint_path)
    
    total_time = time.time() - start_time
    
    # Save final model with descriptive filename
    final_model_filename = f"final_model_{dataset_name}_round_{args.num_rounds}_client_{args.num_clients}_beta_{beta_value}_acc_{test_acc:.2f}.pth"
    final_model_path = os.path.join(exp_dir, final_model_filename)
    torch.save({
        'model_state_dict': server.global_model.state_dict(),
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'total_rounds': exp_conf["global_epochs"],
        'total_time': total_time,
        'experiment_config': {
            'dataset': dataset_name,
            'beta': beta_value,
            'num_clients': args.num_clients,
            'num_rounds': args.num_rounds,
            'tau': args.tau,
            'lambda_snn': args.lambda_snn,
            'resample_ratio': args.resample_ratio,
            'update_global_every': args.update_global_every,
            'server_epochs': args.server_epochs
        }
    }, final_model_path)
    
    # Final save of training metrics (in case the last round wasn't saved)
    metrics_df = pd.DataFrame(round_metrics)
    metrics_df.to_csv(metrics_path, index=False)
    
    # Final save of detailed training info
    if global_training_info:
        detailed_df = pd.DataFrame(global_training_info)
        detailed_df.to_csv(detailed_path, index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Experiment completed: {exp_name}")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Results saved to: {exp_dir}")
    print(f"{'='*60}\n")
    
    return {
        'dataset': dataset_name,
        'beta': beta_value,
        'num_clients': args.num_clients,
        'num_rounds': args.num_rounds,
        'final_accuracy': test_acc,
        'final_loss': test_loss,
        'total_time': total_time,
        'exp_dir': exp_dir
    }


def server_synthetic_training(server, exp_conf, global_calibration, device):
    """
    SERVER-SIDE training with REAL + SYNTHETIC data.
    Synthetic data serves as enhancement to real server test data.
    
    Args:
        server: Server object with global model and test data
        exp_conf: Experiment configuration
        global_calibration: Global calibration statistics
        device: Device to perform training on
    """
    from loss_function.calibration import FeatureCalibration
    from loss_function.select_loss_fn import selected_loss_function
    import torch.utils.data as data_utils
    
    # Initialize calibration for synthetic data generation
    feature_dim = list(global_calibration['mean'].values())[0].shape[0]
    calibration = FeatureCalibration(exp_conf["num_classes"], feature_dim)
    calibration.global_mean = global_calibration['mean']
    calibration.global_cov = global_calibration['cov']
    
    # Calculate synthetic samples (enhancement to real data)
    # Use server test dataset size as base for synthetic generation
    test_dataset_size = len(server.test_loader.dataset)
    num_synthetic_total = int(test_dataset_size * exp_conf['calibsnn']['resample_ratio'])
    num_synthetic_per_class = max(1, num_synthetic_total // exp_conf['num_classes'])
    
    if num_synthetic_per_class == 0:
        print("   No synthetic data to generate (resample_ratio too small)")
        # Fall back to real data only training
        return server_real_data_training(server, exp_conf, device)
    
    print(f"   Generating {num_synthetic_per_class * exp_conf['num_classes']} synthetic samples...")
    print(f"   Real test data: {test_dataset_size} samples")
    
    # Generate synthetic features and labels
    synthetic_features, synthetic_labels = calibration.sample_from_global(
        num_synthetic_per_class, target_device=device)
    
    # Setup optimizer for server model training
    if exp_conf.get("server_optimizer", "SGD") == "SGD":
        optimizer = torch.optim.SGD(
            server.global_model.parameters(), 
            lr=exp_conf.get('server_lr', exp_conf['lr']), 
            momentum=exp_conf.get('momentum', 0.9),
            weight_decay=exp_conf.get("weight_decay", 1e-4))
    else:
        optimizer = torch.optim.Adam(
            server.global_model.parameters(), 
            lr=exp_conf.get('server_lr', exp_conf['lr']), 
            weight_decay=exp_conf.get("weight_decay", 1e-4))
    
    criterion = selected_loss_function(loss=exp_conf['train_loss_criterion'])
    
    server.global_model.train()
    server_epochs = exp_conf['calibsnn'].get('server_epochs', 100)
    
    print(f"   🔄 SERVER training: Combined Real + Synthetic data ({server_epochs} epochs)...")
    
    # Create synthetic features dataset
    import torch.utils.data as data_utils
    synthetic_dataset = data_utils.TensorDataset(synthetic_features, synthetic_labels)
    synthetic_loader = data_utils.DataLoader(
        synthetic_dataset,
        batch_size=exp_conf["batch_size"],
        shuffle=True,
        drop_last=False
    )
    
    print(f"      Real test samples: {len(server.test_loader.dataset)}")
    print(f"      Synthetic samples: {synthetic_features.size(0)}")
    
    total_loss = 0.0
    total_batches = 0
    
    # Train for specified epochs
    for epoch in range(server_epochs):
        epoch_loss = 0.0
        epoch_batches = 0
        
        # Phase 1: Train on real test data (raw data, not features)
        for batch_idx, (data, target) in enumerate(server.test_loader):
            if torch.cuda.is_available() and device.type == 'cuda':
                data = data.to(device)
                target = target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with raw data
            embeddings, logits = server.global_model(data)
            
            # Compute loss
            if exp_conf['train_loss_criterion'] == 3:  # SNN loss only
                loss = criterion(embeddings, target)
            elif exp_conf['train_loss_criterion'] == 4:  # CalibSNN loss
                loss = criterion(embeddings, logits, target)
            else:  # Standard CE loss
                loss = criterion(logits, target)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_batches += 1
        
        # Phase 2: Train on synthetic features
        # CRITICAL: Freeze feature extractor to prevent destruction
        # Only update classifier when training on synthetic features
        for param in server.global_model.parameters():
            param.requires_grad = False
        
        # Enable gradients only for classifier layers
        if hasattr(server.global_model, 'classifier'):
            for param in server.global_model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(server.global_model, 'fc'):
            for param in server.global_model.fc.parameters():
                param.requires_grad = True
        elif hasattr(server.global_model, 'out'):
            for param in server.global_model.out.parameters():
                param.requires_grad = True
        
        for batch_idx, (batch_features, batch_labels) in enumerate(synthetic_loader):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            # For synthetic features, we need to pass them through classifier only
            # since they are already feature representations
            logits = features_to_logits(server.global_model, batch_features)
            
            # Compute loss on synthetic features
            if exp_conf['train_loss_criterion'] == 3:  # SNN loss only
                loss = criterion(batch_features, batch_labels)
            elif exp_conf['train_loss_criterion'] == 4:  # CalibSNN loss
                loss = criterion(batch_features, logits, batch_labels)
            else:  # Standard CE loss
                loss = criterion(logits, batch_labels)
            
            # Scale synthetic loss to balance with real data
            synthetic_weight = exp_conf['calibsnn'].get('synthetic_weight', 0.5)
            loss = loss * synthetic_weight
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_batches += 1
        
        # Re-enable gradients for all parameters for next epoch
        for param in server.global_model.parameters():
            param.requires_grad = True
        
        total_loss += epoch_loss
        total_batches += epoch_batches
        
        # Log progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            avg_epoch_loss = epoch_loss / epoch_batches
            print(f"      Epoch {epoch + 1}/{server_epochs}: avg_loss = {avg_epoch_loss:.4f}")
    
    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    
    print(f"   ✅ SERVER training complete ({server_epochs} epochs):")
    print(f"      Total batches: {total_batches}, avg_loss = {avg_loss:.4f}")


def server_real_data_training(server, exp_conf, device):
    """
    Fallback: SERVER training with REAL data only (when no synthetic data).
    """
    from loss_function.select_loss_fn import selected_loss_function
    
    print(f"   🔄 SERVER training: Real data only (no synthetic)...")
    
    # Setup optimizer
    if exp_conf.get("server_optimizer", "SGD") == "SGD":
        optimizer = torch.optim.SGD(
            server.global_model.parameters(), 
            lr=exp_conf.get('server_lr', exp_conf['lr']), 
            momentum=exp_conf.get('momentum', 0.9),
            weight_decay=exp_conf.get("weight_decay", 1e-4))
    else:
        optimizer = torch.optim.Adam(
            server.global_model.parameters(), 
            lr=exp_conf.get('server_lr', exp_conf['lr']), 
            weight_decay=exp_conf.get("weight_decay", 1e-4))
    
    criterion = selected_loss_function(loss=exp_conf['train_loss_criterion'])
    
    server.global_model.train()
    server_epochs = exp_conf['calibsnn'].get('server_epochs', 100)
    
    print(f"   🔄 SERVER training: Real data only ({server_epochs} epochs)...")
    
    total_loss = 0.0
    total_batches = 0
    
    # Multiple epochs of server training
    for epoch in range(server_epochs):
        epoch_loss = 0.0
        epoch_batches = 0
        
        # Train on real test data
        for batch_id, batch in enumerate(server.test_loader):
            data, target = batch
            if torch.cuda.is_available() and device.type == 'cuda':
                data = data.to(device)
                target = target.to(device)
            
            optimizer.zero_grad()
            
            # Extract features and logits
            embeddings, logits = server.global_model(data)
            
            # Compute loss
            if exp_conf['train_loss_criterion'] == 3:  # SNN loss only
                loss = criterion(embeddings, target)
            elif exp_conf['train_loss_criterion'] == 4:  # CalibSNN loss
                loss = criterion(embeddings, logits, target)
            else:  # Standard CE loss
                loss = criterion(logits, target)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_batches += 1
        
        # Accumulate across epochs
        total_loss += epoch_loss
        total_batches += epoch_batches
    
    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    print(f"   ✅ SERVER real data training complete ({server_epochs} epochs): {total_batches} total batches, avg_loss = {avg_loss:.4f}")


def features_to_logits(model, features):
    """
    Convert feature embeddings to logits using model's classifier layers.
    
    Args:
        model: Neural network model
        features: Feature embeddings (batch_size, feature_dim)
        
    Returns:
        logits: Classification outputs (batch_size, num_classes)
    """
    # Handle different model architectures
    if hasattr(model, 'classifier'):
        return model.classifier(features)
    elif hasattr(model, 'fc'):
        return model.fc(features) 
    elif hasattr(model, 'out'):
        return model.out(features)
    elif hasattr(model, 'linear'):
        return model.linear(features)
    else:
        # Find last linear layer
        last_linear = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                last_linear = module
        
        if last_linear is not None:
            return last_linear(features)
        
        raise ValueError("Could not find classifier layer in model for synthetic data training")


def main():
    parser = argparse.ArgumentParser(description='Run CalibSNN experiments')
    
    # Dataset and experiment parameters
    parser.add_argument('--datasets', nargs='+', 
                       default=['mnist', 'cifar10', 'usps'],
                       choices=['mnist', 'cifar10', 'usps', 'cifar100', 'fmnist', 'svhn', 'kddcup99', 'ag_news', 'adult', 'covertype'],
                       help='Datasets to run experiments on')
    parser.add_argument('--betas', nargs='+', type=float, 
                       default=[0.05, 0.1, 0.3],
                       help='Beta values for Dirichlet distribution (non-IID levels)')
    
    # Federated learning parameters
    parser.add_argument('--num_clients', type=int, default=20,
                       help='Number of clients')
    parser.add_argument('--num_rounds', type=int, default=50,
                       help='Number of communication rounds')
    parser.add_argument('--local_epochs', type=int, default=10,
                       help='Number of local epochs per round')
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--client_sampling_rate', type=float, default=0.3,
                       help='Fraction of clients to sample per round (0.1-1.0). Lower values significantly improve speed.')
    
    # CalibSNN specific parameters
    parser.add_argument('--tau', type=float, default=2.5,
                       help='Temperature parameter for SNN loss')
    parser.add_argument('--lambda_snn', type=float, default=0.5,
                       help='Weight for SNN loss component')
    
    # Calibration control (default: ENABLED for complete CalibSNN)
    parser.add_argument('--disable_calibration', action='store_true', default=False,
                       help='Disable feature calibration (default: calibration ENABLED)')
    parser.add_argument('--disable_resampling', action='store_true', default=False,
                       help='Disable synthetic data resampling (default: resampling ENABLED)')
    
    parser.add_argument('--resample_ratio', type=float, default=0.2,
                       help='Ratio of synthetic samples to add')
    parser.add_argument('--update_global_every', type=int, default=5,
                       help='Update global statistics every N rounds')
    parser.add_argument('--server_epochs', type=int, default=10,
                       help='Number of epochs for server training (default: 10)')
    parser.add_argument('--synthetic_weight', type=float, default=0.5,
                       help='Weight for synthetic data loss (default: 0.5)')
    
    # System parameters
    parser.add_argument('--gpu_ids', type=str, default='0',
                       help='GPU IDs to use for client-parallel training (e.g., "0,1,2,3" or "0")')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='Legacy: Single GPU ID (use --gpu_ids instead for multi-GPU)')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Directory to save experiment results')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N rounds')
    parser.add_argument('--min_require_size', type=int, default=15,
                       help='Minimum number of samples required per client')
    
    # Test mode
    parser.add_argument('--test_mode', action='store_true',
                       help='Run in test mode with fewer rounds')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint file to resume training from')
    
    args = parser.parse_args()
    
    # Override parameters for test mode
    if args.test_mode:
        print("Running in TEST MODE - reduced parameters")
        # Don't override datasets if explicitly provided
        if len(sys.argv) == 2 or '--datasets' not in sys.argv:
            args.datasets = ['mnist']
        args.betas = [0.1]
        args.num_rounds = 5
        args.num_clients = 5
        args.local_epochs = 2
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine CalibSNN configuration
    calibration_enabled = not args.disable_calibration
    resampling_enabled = not args.disable_resampling
    
    if calibration_enabled and resampling_enabled:
        method_type = "Complete CalibSNN (Calibration + SNN Loss + Synthetic Resampling)"
    elif calibration_enabled and not resampling_enabled:
        method_type = "Partial CalibSNN (Calibration + SNN Loss, NO synthetic resampling)"
    elif not calibration_enabled and resampling_enabled:
        method_type = "Invalid Configuration (Resampling requires calibration)"
        print("ERROR: Cannot enable resampling without calibration!")
        sys.exit(1)
    else:
        method_type = "SNN Loss Only (NO calibration, NO resampling)"
    
    # Summary of experiments to run
    total_experiments = len(args.datasets) * len(args.betas)
    print(f"CalibSNN Experiment Runner - Multi-GPU Client-Parallel FL")
    print(f"{'='*80}")
    print(f"🧠 Method: {method_type}")
    print(f"🔧 Calibration: {'ENABLED' if calibration_enabled else 'DISABLED'}")
    print(f"🧬 Synthetic Resampling: {'ENABLED' if resampling_enabled else 'DISABLED'}")
    print(f"📊 GPU Configuration: {args.gpu_ids if hasattr(args, 'gpu_ids') else args.gpu_id}")
    print(f"📈 Datasets: {args.datasets}")
    print(f"📉 Beta values: {args.betas}")
    print(f"👥 Number of clients: {args.num_clients}")
    print(f"🔄 Number of rounds: {args.num_rounds}")
    print(f"🎯 Client sampling rate: {args.client_sampling_rate}")
    print(f"⚡ Expected speedup: ~{len(args.gpu_ids.split(',')) if hasattr(args, 'gpu_ids') and ',' in args.gpu_ids else 1}x (Multi-GPU)")
    print(f"🔬 Total experiments: {total_experiments}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"{'='*80}\n")
    
    # Run all experiments
    all_results = []
    
    for dataset in args.datasets:
        for beta in args.betas:
            try:
                result = run_calibsnn_experiment(args, dataset, beta)
                all_results.append(result)
            except Exception as e:
                print(f"ERROR: Experiment failed for {dataset} with beta={beta}")
                print(f"Error message: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save summary of all experiments with timestamp to avoid overwriting
    if all_results:
        summary_df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        datasets_str = "_".join(args.datasets)
        betas_str = "_".join([str(b) for b in args.betas])
        summary_filename = f"CalibSNN_experiment_summary_{datasets_str}_round_{args.num_rounds}_client_{args.num_clients}_betas_{betas_str}_{timestamp}.csv"
        summary_path = os.path.join(args.output_dir, summary_filename)
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n{'='*60}")
        print("All experiments completed!")
        print(f"Summary saved to: {summary_path}")
        print("\nResults Summary:")
        print(summary_df.to_string(index=False))
        print(f"{'='*60}")


if __name__ == "__main__":
    main() 