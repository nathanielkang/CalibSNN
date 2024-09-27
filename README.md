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

