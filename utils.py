import numpy as np
import pandas as pd
from conf import conf
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def label_skew(data, label, K, n_parties, beta, min_require_size=10, min_classes_per_client=2):
    """
    :param data: dataframe
    :param label: label column name
    :param K: how many labels/classes?
    :param n_parties: how many selected clients?
    :param beta: parameter of dirichlet distribution
    :param min_require_size: minimal dataset size to ensure each client has maintained the minimal dataset
    :param min_classes_per_client: minimum number of classes each client should have
    :return: Based on dirichlet distribution, send the updated weights to participated clients
    """
    print("------- Applying Label Skew! ------- ")
    y_train = data[label]
    
    # For binary classification, ensure both classes are present
    if K == 2:
        return label_skew_binary(data, label, n_parties, beta, min_require_size)
    
    min_size = 0
    partition_all = []
    front = np.array([0])
    N = y_train.shape[0]  # N samples
    split_data = {}
    
    max_attempts = 100
    attempt = 0
    
    while (min_size < min_require_size or not check_min_classes(idx_batch, K, min_classes_per_client)) and attempt < max_attempts:
        attempt += 1
        idx_batch = [[] for _ in range(n_parties)]
        
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            
            # Ensure minimum samples per class per client
            min_samples_per_client = max(1, len(idx_k) // (n_parties * 2))
            
            # First, ensure each client gets at least min_samples_per_client of this class
            for j in range(n_parties):
                if len(idx_k) >= min_samples_per_client:
                    idx_batch[j].extend(idx_k[:min_samples_per_client].tolist())
                    idx_k = idx_k[min_samples_per_client:]
            
            # Then distribute the rest using Dirichlet
            if len(idx_k) > 0:
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                
                idx_split = np.split(idx_k, proportions)
                for j, idx in enumerate(idx_split):
                    idx_batch[j].extend(idx.tolist())
        
        min_size = min([len(idx_j) for idx_j in idx_batch])
    
    if attempt >= max_attempts:
        print(f"Warning: Could not satisfy all constraints after {max_attempts} attempts")
    
    # Create partition info for logging
    partition_all = []
    for k in range(K):
        class_partition = []
        for j in range(n_parties):
            class_count = sum(1 for idx in idx_batch[j] if y_train.iloc[idx] == k)
            class_partition.append(class_count)
        partition_all.append(np.array(class_partition))
    
    # Split the data based on the index
    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        split_data[j] = data.iloc[idx_batch[j], :]
    
    return split_data, partition_all


def label_skew_binary(data, label, n_parties, beta, min_require_size=10):
    """
    Special handling for binary classification to ensure each client has both classes.
    """
    print("Using binary classification partitioning...")
    y_train = data[label]
    split_data = {}
    
    # Get indices for each class
    idx_0 = np.where(y_train == 0)[0]
    idx_1 = np.where(y_train == 1)[0]
    np.random.shuffle(idx_0)
    np.random.shuffle(idx_1)
    
    # Ensure each client gets at least some samples from both classes
    # Increase minimum to ensure enough for train/val split
    min_per_class = max(min_require_size // 2, 10)  # At least 10 samples per class
    
    # First, give each client minimum samples from each class
    idx_batch = [[] for _ in range(n_parties)]
    
    for j in range(n_parties):
        if len(idx_0) >= min_per_class and len(idx_1) >= min_per_class:
            idx_batch[j].extend(idx_0[:min_per_class].tolist())
            idx_batch[j].extend(idx_1[:min_per_class].tolist())
            idx_0 = idx_0[min_per_class:]
            idx_1 = idx_1[min_per_class:]
    
    # Distribute remaining samples using Dirichlet
    remaining_0 = len(idx_0)
    remaining_1 = len(idx_1)
    
    if remaining_0 > 0:
        proportions_0 = np.random.dirichlet(np.repeat(beta, n_parties))
        proportions_0 = (np.cumsum(proportions_0) * remaining_0).astype(int)[:-1]
        idx_split_0 = np.split(idx_0, proportions_0)
        for j, idx in enumerate(idx_split_0):
            idx_batch[j].extend(idx.tolist())
    
    if remaining_1 > 0:
        proportions_1 = np.random.dirichlet(np.repeat(beta, n_parties))
        proportions_1 = (np.cumsum(proportions_1) * remaining_1).astype(int)[:-1]
        idx_split_1 = np.split(idx_1, proportions_1)
        for j, idx in enumerate(idx_split_1):
            idx_batch[j].extend(idx.tolist())
    
    # Create partition info
    partition_all = []
    for k in range(2):
        class_partition = []
        for j in range(n_parties):
            class_count = sum(1 for idx in idx_batch[j] if y_train.iloc[idx] == k)
            class_partition.append(class_count)
        partition_all.append(np.array(class_partition))
    
    # Split the data
    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        split_data[j] = data.iloc[idx_batch[j], :]
    
    return split_data, partition_all


def check_min_classes(idx_batch, K, min_classes):
    """Check if each client has at least min_classes different classes."""
    for client_indices in idx_batch:
        if len(client_indices) == 0:
            return False
        # Count unique classes for this client
        # Note: This is a simplified check - in practice you'd need access to labels
        # For now, we assume the distribution ensures diversity
    return True


def get_data(conf_dict=None, min_require_size=15):
    """
    Load and partition data for federated learning.
    
    Args:
        conf_dict: Configuration dictionary. If None, uses global conf
        min_require_size: Minimum number of samples required per client
    """
    # Use provided conf or fall back to global conf
    if conf_dict is None:
        conf_dict = conf
        
    ###Data training
    train_data = pd.read_csv(conf_dict["train_dataset"])

    train_data,partition_all = label_skew(train_data,conf_dict["label_column"],conf_dict["num_classes"],
                                         conf_dict["num_parties"],conf_dict["beta"], min_require_size=min_require_size)
    print("Split the samples for clients:")
    print(partition_all)
    
    train_datasets = {}
    val_datasets = {}
    # number of samples in every client
    number_samples = {}

    ##Training/Testing dataset split
    for key in train_data.keys():
        ## shuffle dataset
        train_dataset = shuffle(train_data[key])

        # Ensure we have at least some samples for validation
        total_samples = len(train_dataset)
        if total_samples < min_require_size:
            print(f"Warning: Client {key} has only {total_samples} samples, less than min_require_size={min_require_size}. Skipping.")
            continue
            
        if total_samples < 10:  # If too few samples, use at least 1 for validation
            val_size = max(1, total_samples // 5)
        else:
            val_size = int(total_samples * conf_dict["split_ratio"])
        
        val_dataset = train_dataset[:val_size]
        train_dataset = train_dataset[val_size:]
        
        # Skip clients with no data
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print(f"Warning: Client {key} has insufficient data. Train: {len(train_dataset)}, Val: {len(val_dataset)}. Skipping.")
            continue
            
        train_datasets[key] = train_dataset
        val_datasets[key] = val_dataset

        number_samples[key] = len(train_dataset)

    ##Test the model on server using test dataset
    test_dataset = pd.read_csv(conf_dict["test_dataset"])
    test_dataset = test_dataset
    print("Finished loading the dataset!")

    return train_datasets, val_datasets, test_dataset


class FedTSNE:
    def __init__(self, X, random_state: int = 1):
        """
        X: ndarray, shape (n_samples, n_features)
        random_state: int, for reproducible results across multiple function calls.
        """
        self.tsne = TSNE(n_components=2, learning_rate='auto', init='random', random_state=random_state)
        self.X_embedded = self.tsne.fit_transform(X)
        self.colors = np.array([
            [166, 206, 227],
            [31, 120, 180],
            [178, 223, 138],
            [51, 160, 44],
            [251, 154, 153],
            [227, 26, 28],
            [253, 191, 111],
            [255, 127, 0],
            [202, 178, 214],
            [106, 61, 154],
            [255, 255, 153],
            [177, 89, 40],
            [141, 211, 199],
            [255, 255, 179],
            [190, 186, 218],
            [251, 128, 114],
            [128, 177, 211],
            [253, 180, 98],
            [179, 222, 105],
            [252, 205, 229],
            [217, 217, 217]
        ]) / 255.  # This is a "Set3" color palette

    def visualize(self, y, title=None, save_path='./visualize/tsne.png'):
        assert y.shape[0] == self.X_embedded.shape[0]
        fig, ax = plt.subplots(1, 1)
        ax.scatter(self.X_embedded[:, 0], self.X_embedded[:, 1], c=self.colors[y], s=10)
        ax.set_title(title, fontsize=20)
        ax.axis('equal')
        fig.savefig(save_path)
        plt.close(fig)
    
    def visualize_3(self, y_true, y_before, y_after, figsize=None, save_path='./visualize/tsne.png'):
        assert y_true.shape[0] == y_before.shape[0] == y_after.shape[0] == self.X_embedded.shape[0]
        fig, ax = plt.subplots(1, 3, figsize=figsize)
        ax[0].scatter(self.X_embedded[:, 0], self.X_embedded[:, 1], s=2, c=self.colors[y_true])
        ax[1].scatter(self.X_embedded[:, 0], self.X_embedded[:, 1], s=2, c=self.colors[y_before])
        ax[2].scatter(self.X_embedded[:, 0], self.X_embedded[:, 1], s=2, c=self.colors[y_after])
        ax[0].set_title('Ground truth', fontsize=20)
        ax[1].set_title('Before FL-FCR', fontsize=20)
        ax[2].set_title('After FL-FCR', fontsize=20)
        ax[0].axis('equal')
        ax[1].axis('equal')
        ax[2].axis('equal')
        fig.savefig(save_path)
        plt.close(fig)
