# CalibSNN: Re-sampling Calibrated SNN Loss for Non-IID Data in Federated Learning

**CalibSNN** is a robust approach designed to handle the challenges posed by non-IID data distributions in Federated Learning. By integrating re-sampling techniques with the Soft Nearest Neighbor (SNN) loss, the model improves calibration, reduces data bias, and enhances feature representation in federated learning environments.

## Features
- **Addresses Data Imbalance and Non-IID Distributions**: Enhances federated learning models by mitigating data heterogeneity across clients.
- **Implements SNN Loss**: Utilizes Soft Nearest Neighbor loss for better model calibration and robustness.
- **Supports Multi-Modal Datasets**: Compatible with both tabular and image-based data.
- **Visualization Tools**: Provides tools for visualizing model performance and training metrics.



## Installation

To set up the environment and install the required dependencies, follow these steps:

```bash
# Clone the repository
git clone https://github.com/nathanielkang/CalibSNN.git
cd CalibSNN

# Install dependencies
pip install -r requirements.txt
``` 

## Usage

### 1. Preprocess Data

The `data_process.py` and `kaggle_data_process.py` scripts handle data loading and preprocessing. Ensure your dataset is placed in the appropriate directory specified in `conf.py`.

```bash
python data_process.py
```

### 2. Training the Model

```bash
python main.py
```


# Configuration

The package comes with a `conf.py` file that allows users to modify various configurations for training and evaluation. Below are some of the key parameters you can adjust:

## General Parameters
dataset_used: Specifies the dataset (e.g., mnist, cifar10, cifar100, etc.).
model_name: Defines the model architecture (e.g., mlp, cnn).
classification_type: Defines the type of classification (binary or multi).

## Loss Criteria
- **train_loss_criterion: Defines the loss function for training.
  1: BinaryCrossEntropy
  2: CategoricalCrossEntropy
  3: FedLCalibratedLoss
  4: ContrastiveLoss

## Contrastive Learning Settings
- **train_contrastive_learning: Set to True to enable contrastive learning during training.
- **eval_contrastive_learning: Set to True to enable contrastive learning during evaluation.

## Federated Learning Parameters
- **client_optimizer: Optimizer for client training (Adam or SGD).
- **re_train_optimizer: Optimizer for retraining models (Adam or SGD).
- **global_epochs: Number of global epochs for FL training.
- **local_epochs: Number of local epochs per client.

## Data Parameters
- **data_type: Specifies the type of data (image, tabular).
- **batch_size: Defines the batch size for training.
- **beta: Specifies the value of the Dirichlet distribution for non-IID data.
- **split_ratio: Ratio of validation data split for local clients.


## Model Saving and Retraining
- **model_dir: Directory where the model will be saved.
- **model_file: Filename of the saved model.
- **retrain_model_file: Filename for the retrained model.

## Other Key Parameters
- **lr: Learning rate for optimization.
- **momentum: Momentum for the optimizer (if applicable).
- **weight_decay: Weight decay for regularization.
- **gamma: Hyperparameter controlling the distribution of synthetic data.












