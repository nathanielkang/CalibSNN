import argparse
import torch

from conf import conf
from fedavg.server import Server
from fedavg.client import Client
from fedavg.models import CNN_Model,MLP
# from utils import get_cifar10, FedTSNE
from utils import get_data, FedTSNE
import os

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


import argparse
import torch
from conf import conf
from fedavg.server import Server
from fedavg.client import Client
from fedavg.models import CNN_Model, MLP
from utils import get_data, FedTSNE
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Define paths for before and after calibration models, and the TSNE result image
    before_calibration_model_path = './save_model/' + conf['dataset_used'] + '_' + 'model.pth'
    after_calibration_model_path = './save_model/' + conf['dataset_used'] + '_' + 'retrained_model.pth'
    saved_tsne_path = './visualize/' + conf['dataset_used'] + '_' + 'tsne.png'
    
    # Ensure save directories exist
    save_dir = './visualize/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_before_calibration', default=before_calibration_model_path, type=str, help='path to model before calibration')
    parser.add_argument('--model_after_calibration', default=after_calibration_model_path, type=str, help='path to model after calibration')
    parser.add_argument('--random_state', default=1, type=int, help='random state for tsne')
    parser.add_argument('--save_path', default=saved_tsne_path, type=str, help='path to save tsne result')
    args = parser.parse_args()

    # Load dataset
    train_datasets, val_datasets, test_dataset = get_data()

    # Model definition
    if conf['model_name'] == "mlp":
        model = MLP(2352, 512, conf["num_classes"])
        print("-----------Model for MLP selected!-----------")
        if conf['dataset_used'] == "kaggle" and conf['kaggle']['dataset_used'] == "adult_income":
            model = MLP(n_feature=14, n_hidden=512, n_output=1)
    elif conf['model_name'] == "cnn":
        model = CNN_Model()

    # Move model to GPU if available
    if torch.cuda.is_available():
        model.cuda()

    # Initialize server
    server = Server(conf, model, test_dataset)
    print('Start TSNE...')

    # Load model before calibration
    server.global_model.load_state_dict(torch.load(args.model_before_calibration))
    tsne_features, tsne_true_labels, tsne_before_labels = server.get_feature_label()

    # Load model after calibration
    server.global_model.load_state_dict(torch.load(args.model_after_calibration))
    _, _, tsne_after_labels = server.get_feature_label()

    # Perform TSNE visualization
    fed_tsne = FedTSNE(tsne_features.detach().cpu().numpy(), random_state=args.random_state)
    fed_tsne.visualize_3(tsne_true_labels.detach().cpu().numpy(),
                         tsne_before_labels.detach().cpu().numpy(),
                         tsne_after_labels.detach().cpu().numpy(),
                         figsize=(15, 3), save_path=args.save_path)
    print('TSNE done.')

    print("Start Confusion Matrix...")

    # Calculate confusion matrix for binary classification
    if conf["num_classes"] == 2:
        # Convert probabilities to binary predictions (0 or 1)
        binary_before_predictions = (tsne_before_labels >= 0.5).int().cpu().numpy()
        binary_after_predictions = (tsne_after_labels >= 0.5).int().cpu().numpy()

        # Create confusion matrices
        cm_before = confusion_matrix(tsne_true_labels.cpu().numpy(), binary_before_predictions)
        cm_after = confusion_matrix(tsne_true_labels.cpu().numpy(), binary_after_predictions)

        # Visualize and save confusion matrices using seaborn
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.heatmap(cm_before, annot=True, fmt="d", cmap="RdPu", cbar=False)
        plt.title("Confusion Matrix Before FL-FCR", fontsize=20)
        plt.xlabel("Predicted label", fontsize=18)
        plt.ylabel("True label", fontsize=18)

        plt.subplot(1, 2, 2)
        sns.heatmap(cm_after, annot=True, fmt="d", cmap="RdPu", cbar=False)
        plt.title("Confusion Matrix After FL-FCR", fontsize=20)
        plt.xlabel("Predicted label", fontsize=18)
        plt.ylabel("True label", fontsize=18)
        plt.savefig(os.path.join(save_dir, f'confusion_matrix_{conf["dataset_used"]}.png'))
        plt.close()

    # Calculate confusion matrix for multi-class classification
    else:
        multiclass_before_predictions = tsne_before_labels.cpu().numpy()
        multiclass_after_predictions = tsne_after_labels.cpu().numpy()

        cm_before = confusion_matrix(tsne_true_labels.cpu().numpy(), multiclass_before_predictions)
        cm_after = confusion_matrix(tsne_true_labels.cpu().numpy(), multiclass_after_predictions)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.heatmap(cm_before, annot=True, fmt="d", cmap="OrRd", cbar=False)
        plt.title("Confusion Matrix Before FL-FCR", fontsize=20)
        plt.xlabel("Predicted label", fontsize=18)
        plt.ylabel("True label", fontsize=18)

        plt.subplot(1, 2, 2)
        sns.heatmap(cm_after, annot=True, fmt="d", cmap="OrRd", cbar=False)
        plt.title("Confusion Matrix After FL-FCR", fontsize=20)
        plt.xlabel("Predicted label", fontsize=18)
        plt.ylabel("True label", fontsize=18)
        plt.savefig(os.path.join(save_dir, f'confusion_matrix_{conf["dataset_used"]}.png'))
        plt.close()

    print("Confusion Matrix Done.")
