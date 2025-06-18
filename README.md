# CalibSNN - Minimal Implementation

This is a minimal implementation of **CalibSNN** (Calibrated Soft Nearest Neighbor Loss) for Federated Learning with non-IID data, as described in the paper "Re-sampling Calibrated SNN Loss: A Robust Approach to Non-IID Data in Federated Learning".

## 🚀 Quick Start

### Prerequisites

Install required dependencies:
```bash
pip install torch torchvision pandas numpy scikit-learn
```

### Essential Files Structure

```
CalibSNN/
├── main.py                      # Main federated learning runner
├── conf.py                      # Configuration and hyperparameters
├── run_CalibSNN.py             # Experiment runner script
├── requirements.txt             # Dependencies
├── utils.py                     # Utility functions
├── data_process.py              # Data processing utilities
├── prepare_data.py              # Data preparation
│
├── loss_function/               # Core CalibSNN components
│   ├── snn_loss.py             # SNN loss implementation
│   ├── calibration.py          # Feature calibration
│   ├── calibrated_loss.py      # Combined loss wrapper
│   └── select_loss_fn.py       # Loss function selector
│
└── fedavg/                     # Federated learning framework
    ├── client_calibsnn.py      # CalibSNN client
    ├── server.py               # FL server
    ├── models.py               # Neural network models
    └── datasets.py             # Dataset handling
```

## 📋 How to Run Experiments

### Option 1: Simple Run (using main.py)
```bash
python main.py
```

### Option 2: Experimental Run (using run_CalibSNN.py)
```bash
# Run on all datasets with default parameters
python run_CalibSNN.py

# Run on specific datasets
python run_CalibSNN.py --datasets mnist cifar10

# Run with custom parameters
python run_CalibSNN.py \
    --datasets mnist \
    --betas 0.05 0.1 0.3 \
    --num_clients 20 \
    --num_rounds 50 \
    --tau 2.5 \
    --lambda_snn 0.5
```

### Option 3: Quick Test Mode
```bash
python run_CalibSNN.py --test_mode
```

## ⚙️ Key Configuration Parameters

Edit `conf.py` to customize:

### CalibSNN Specific
- `'--tau': 2.5` - Temperature parameter for SNN loss
- `'lambda_snn': 0.5` - Weight for SNN loss component
- `'calibsnn.enable_calibration': True` - Enable feature calibration
- `'calibsnn.enable_resampling': True` - Enable re-sampling
- `'calibsnn.resample_ratio': 0.2` - Ratio of synthetic samples

### Federated Learning
- `'num_parties': 20` - Number of clients
- `'global_epochs': 50` - Communication rounds
- `'local_epochs': 10` - Local training epochs
- `'beta': 0.05` - Non-IID level (lower = more heterogeneous)

### Dataset & Model
- `'dataset_used': "cifar10"` - Dataset (mnist, cifar10, usps)
- `'model_name': "cnn"` - Model architecture
- `'batch_size': 1024` - Batch size
- `'lr': 0.01` - Learning rate

## 📊 Supported Datasets

Prepare datasets using:
```bash
# Prepare specific dataset
python prepare_data.py --datasets mnist

# Prepare multiple datasets  
python prepare_data.py --datasets mnist cifar10 usps
```

Supported datasets:
- **MNIST**: Handwritten digits (28x28 grayscale)
- **CIFAR-10**: Natural images (32x32 color)
- **USPS**: Postal handwritten digits (16x16 grayscale)

## 🔧 Core CalibSNN Components

### 1. SNN Loss (`loss_function/snn_loss.py`)
Implements the Soft Nearest Neighbor loss:
- Encourages intra-class compactness
- Promotes inter-class separation
- Temperature parameter τ controls concentration

### 2. Feature Calibration (`loss_function/calibration.py`)
Handles distribution alignment:
- Computes local mean μ_{k,c} and covariance Σ_{k,c}
- Aggregates global statistics μ_c and Σ_c
- Calibrates features to match global distribution

### 3. CalibSNN Client (`fedavg/client_calibsnn.py`)
Integrates calibration with local training:
- Computes local statistics
- Applies feature calibration
- Trains with combined CE + SNN loss

## 📈 Output Structure

Results are saved in:
```
experiments/
└── CalibSNN_[dataset]_beta[value]_C[clients]_R[rounds]_[timestamp]/
    ├── config.json              # Experiment configuration
    ├── round_metrics.csv        # Per-round metrics
    └── final_model.pth          # Trained model
```

## 🔍 Key Differences from Standard FL

| Aspect | Standard FL | CalibSNN |
|--------|-------------|----------|
| Loss Function | Cross-entropy only | CE + SNN loss |
| Data Handling | Raw local data | Calibrated features |
| Aggregation | Model parameters only | Parameters + statistics |
| Non-IID Handling | Limited | Feature-level calibration |

## 🎯 Expected Results

CalibSNN typically achieves:
- **Better accuracy** under non-IID conditions
- **Faster convergence** (2-4x speedup)
- **More robust features** across heterogeneous clients
- **Improved performance** especially at low β values (high heterogeneity)

## 📝 Citation

If you use this code, please cite:
```bibtex
@article{calibsnn2024,
  title={Re-sampling Calibrated SNN Loss: A Robust Approach to Non-IID Data in Federated Learning},
  author={Kang, Nathaniel and Im, Jongho},
  year={2024}
}
```

## 🛠️ Troubleshooting

### Common Issues

1. **Numerical Instability**: Increase `tau` parameter (e.g., 3.0)
2. **Out of Memory**: Reduce `batch_size` or `num_clients`
3. **Slow Training**: Reduce `local_epochs` or disable resampling
4. **NaN Values**: Decrease `lambda_snn` (e.g., 0.1)

### Quick Fixes
```bash
# Conservative parameters for stability
python run_CalibSNN.py --tau 3.0 --lambda_snn 0.1 --learning_rate 0.005

# CPU-only mode
python run_CalibSNN.py --gpu_id -1

# Quick test
python run_CalibSNN.py --test_mode --num_rounds 10
``` 