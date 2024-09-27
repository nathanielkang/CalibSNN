import torch
import torchvision
from torchvision.utils import save_image
import torchvision.transforms as transforms
import os
import pandas as pd
from conf import conf
import subprocess
import numpy as np
from subprocess import CalledProcessError
from kaggle_data_process import adult_income_dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder
from collections import Counter

def save_img(loader, is_train, target_dir):
    if is_train:
        target_dir = os.path.join(target_dir, 'train')
        index_file = os.path.join(target_dir,'train.csv')
    else:
        target_dir = os.path.join(target_dir, 'test')
        index_file = os.path.join(target_dir, 'test.csv')

    os.makedirs(target_dir, exist_ok=True)

    num = 0

    index_fname  = []

    index_label = []

    for _, batch_data in enumerate(loader):
        data, label = batch_data
        for d,l in zip(data, label):

            #Generate pic and save it into the directory
            result_dir = os.path.join(target_dir, str(l.item()))
            if not os.path.exists(result_dir):
                os.makedirs(result_dir,exist_ok=True)

            #Create image/file names
            file = os.path.join(result_dir, "{0}-{1}.png".format(l.item(), num))

            index_fname.append(file)
            index_label.append(l.item())

            #save images
            save_image(d.data, file)
            num += 1

    #Save index
    index = pd.DataFrame({
        conf["data_column"]:index_fname,
        conf["label_column"]:index_label
    })
    index.to_csv(index_file, index=False)



def delete_data(data_dir):
    # Check if .data directory exists
    if os.path.exists(data_dir):
        print("Deleting existing ./data directory...")
        try:
            # Remove the existing .data directory and its contents
            import shutil
            shutil.rmtree(data_dir)
        except Exception as e:
            print(f"Error deleting .data directory: {e}")


def process_MNIST(data_dir, target_dir):
    """
    :param data_dir: directory
    :param target_dir: taget directory (after processing)
    :return:
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),##default: 32,32
        transforms.ToTensor()
    ]) 
    trainset = torchvision.datasets.MNIST(root=data_dir, train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.MNIST(root=data_dir, train=False,
                                           download=False, transform=transform)

    train_loader =  torch.utils.data.DataLoader(trainset, batch_size=64,shuffle=True)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=64,shuffle=True)

    save_img(train_loader, is_train=True, target_dir=target_dir)
    save_img(test_loader,is_train=False,target_dir=target_dir)
    print("MNIST data process done !")


def process_FashionMNIST(data_dir, target_dir):
    """
    Process Fashion-MNIST dataset and save images to target directories (train/test).
    """
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    trainset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    save_img(train_loader, is_train=True, target_dir=target_dir)
    save_img(test_loader, is_train=False, target_dir=target_dir)
    print("Fashion-MNIST data processing done!")



def process_CIFAR10(data_dir, target_dir):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # CIFAR-10 images are 32x32
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                           download=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    save_img(train_loader, is_train=True, target_dir=target_dir)
    save_img(test_loader, is_train=False, target_dir=target_dir)
    print("CIFAR-10 data process done !")


def filter_classes(data, targets, selected_classes):
    """
    Filter the dataset to only include samples from the selected classes.

    :param data: The dataset images (numpy array)
    :param targets: The dataset labels (list or numpy array)
    :param selected_classes: A list of class indices to retain
    :return: Filtered dataset and labels
    """
    # Convert targets to a numpy array if it's a list
    targets = np.array(targets)

    # Create a mask to filter out the selected classes
    mask = np.isin(targets, selected_classes)

    # Apply the mask to data and targets
    filtered_data = data[mask]
    filtered_targets = targets[mask]

    # Create a mapping from the selected classes to new labels (0, 1, ..., len(selected_classes)-1)
    class_mapping = {orig: new for new, orig in enumerate(selected_classes)}
    mapped_targets = np.array([class_mapping[t] for t in filtered_targets])

    return filtered_data, mapped_targets


def process_CIFAR100(data_dir, target_dir):
    """
    Process a subset of CIFAR-100 with 10 classes.
    """
    selected_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)

    # Filter the dataset to only include selected classes
    trainset.data, trainset.targets = filter_classes(trainset.data, trainset.targets, selected_classes)
    testset.data, testset.targets = filter_classes(testset.data, testset.targets, selected_classes)

    # DataLoader for the training and testing data
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    save_img(train_loader, is_train=True, target_dir=target_dir)
    save_img(test_loader, is_train=False, target_dir=target_dir)
    print("CIFAR-100 data processing done!")



def process_USPS(data_dir, target_dir):
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    trainset = torchvision.datasets.USPS(root=data_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.USPS(root=data_dir, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    save_img(train_loader, is_train=True, target_dir=target_dir)
    save_img(test_loader, is_train=False, target_dir=target_dir)
    print("USPS data processing done!")

def process_SVHN(data_dir, target_dir):
    """
    Process the SVHN dataset and save images to the specified directories (train/test).
    
    :param data_dir: Directory for raw data
    :param target_dir: Directory where processed images will be saved
    """
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    
    # Load the SVHN dataset
    trainset = torchvision.datasets.SVHN(root=data_dir, split='train', download=True, transform=transform)
    testset = torchvision.datasets.SVHN(root=data_dir, split='test', download=True, transform=transform)
    
    # Create DataLoader for train and test data
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    # Save the images and labels for train and test sets
    save_img(train_loader, is_train=True, target_dir=target_dir)
    save_img(test_loader, is_train=False, target_dir=target_dir)
    
    print("SVHN data processing done!")


def process_KAGGLE(target_dir):
    os.environ['KAGGLE_USERNAME'] = conf['kaggle']['kaggle_username']
    os.environ['KAGGLE_KEY'] = conf['kaggle']['kaggle_api_key']
    dataset_name = conf['kaggle']['kaggle_dataset_download']

    try:
        command = f"kaggle datasets download -d {dataset_name} -p {target_dir} --unzip"
        subprocess.run(command, shell=True, check=True)
        print(f"Dataset '{dataset_name}' downloaded successfully.")
    except CalledProcessError as e:
        print(f"Error: {e}")
        print("Please check your Kaggle API credentials and dataset link.")

# CIFAR100 V2 - u of toronto site down
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class CIFAR100Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img)  # Convert to PIL Image
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

def filter_classes(data, targets, selected_classes):
    """
    Filter the dataset to only include samples from the selected classes.

    :param data: The dataset images (numpy array)
    :param targets: The dataset labels (list or numpy array)
    :param selected_classes: A list of class indices to retain
    :return: Filtered dataset and labels
    """
    # Convert targets to a numpy array if it's a list
    targets = np.array(targets)

    # Create a mask to filter out the selected classes
    mask = np.isin(targets, selected_classes)

    # Apply the mask to data and targets
    filtered_data = data[mask]
    filtered_targets = targets[mask]

    # Create a mapping from the selected classes to new labels (0, 1, ..., len(selected_classes)-1)
    class_mapping = {orig: new for new, orig in enumerate(selected_classes)}
    mapped_targets = np.array([class_mapping[t] for t in filtered_targets])

    return filtered_data, mapped_targets

def process_CIFAR100(data_dir, target_dir):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # CIFAR-100 images are 32x32
        transforms.ToTensor()
    ])
    selected_classes = [10,11,12,13,14,15,16,17,18,19]
    train_data_dict = unpickle(os.path.join(data_dir, 'train'))
    test_data_dict = unpickle(os.path.join(data_dir, 'test'))

    # Load the meta data to get class names if needed
    meta_data_dict = unpickle(os.path.join(data_dir, 'meta'))
    fine_label_names = meta_data_dict[b'fine_label_names']

    # Extract data and labels from the dictionaries
    train_data = train_data_dict[b'data']
    train_labels = train_data_dict[b'fine_labels']
    test_data = test_data_dict[b'data']
    test_labels = test_data_dict[b'fine_labels']

    # Reshape data to (N, 32, 32, 3)
    train_data = train_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  
    test_data = test_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # Filter the dataset to only include selected classes
    train_data, train_labels = filter_classes(train_data, train_labels, selected_classes)
    test_data, test_labels = filter_classes(test_data, test_labels, selected_classes)

    # Create custom PyTorch datasets
    trainset = CIFAR100Dataset(train_data, train_labels, transform=transform)
    testset = CIFAR100Dataset(test_data, test_labels, transform=transform)

    # DataLoader for the training and testing data
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
    test_loader = DataLoader(testset, batch_size=64, shuffle=True)

    # Save the images (assuming you have a function to save them)
    save_img(train_loader, is_train=True, target_dir=target_dir)
    save_img(test_loader, is_train=False, target_dir=target_dir)

    print("CIFAR-100 data processing done!")
    
    
#v2 to process cifar10 beacuse u of toronto is down

class CIFAR10DatasetFromPixels(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        Initializes the CIFAR-10 dataset with pixel data and labels.
        
        Args:
            data (numpy.ndarray): The pixel data, shape [N, 3072].
            labels (numpy.ndarray): The labels, shape [N].
            transform (callable, optional): The transformation to apply to images.
        """
        self.data = data
        self.labels = labels
        self.transform = transform

        # Reshape the flat pixel data into (N, 32, 32, 3) images
        self.data = self.data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # Reshape to (N, 32, 32, 3)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img.astype('uint8'))  # Convert to PIL Image
        label = int(self.labels[idx])

        if self.transform:
            img = self.transform(img)

        return img, label

def process_CIFAR10(data_dir, target_dir, test_size=0.2):
    """
    Processes CIFAR-10 dataset from a single CSV file, splits it into training and test sets,
    and saves the processed images.

    Args:
        data_dir (str): The directory containing the CIFAR-10 CSV files.
        target_dir (str): The directory to save the processed data.
        test_size (float): The fraction of data to be used for testing.
    """
    # Define the transformations (resizing and normalization for CIFAR-10)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1] range
    ])

    # Path to the train.csv file
    csv_file = os.path.join(data_dir, 'train.csv')

    # Load the CSV file (assumes first column is the label, and the rest are pixel values)
    data_info = pd.read_csv(csv_file)

    # Extract labels (first column)
    labels = data_info['label'].values  # Should be integers between 0 and 9
    print("Unique labels in dataset:", np.unique(labels))

    # Extract pixel data (remaining columns)
    data = data_info.iloc[:, 1:].values  # Pixel data in shape [N, 3072]
    if data.shape[1] != 3072:
        raise ValueError(f"Expected 3072 pixels per image, but got {data.shape[1]}.")

    # Split the data into training and test sets
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=test_size, random_state=42
    )

    # Create datasets for training and testing
    trainset = CIFAR10DatasetFromPixels(train_data, train_labels, transform=transform)
    testset = CIFAR10DatasetFromPixels(test_data, test_labels, transform=transform)

    # DataLoader for training and testing data
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False)  # No shuffle for test data

    # Save images (this function needs to be implemented or replaced with your actual image saving method)
    save_img(train_loader, is_train=True, target_dir=target_dir)
    save_img(test_loader, is_train=False, target_dir=target_dir)

    print("CIFAR-10 data processing done!")

def process_CINIC10(data_dir, target_dir):
    """
    Process the CINIC-10 dataset and save images to the specified directories (train/test/valid).
    
    :param data_dir: Directory where the CINIC-10 dataset is located.
    :param target_dir: Directory where processed images will be saved.
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Ensure images are resized to 32x32
        transforms.ToTensor()         # Convert to tensor
    ])
    
    # Paths for train, validation, and test sets
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')
    
    # Load the dataset using ImageFolder
    trainset = ImageFolder(root=train_dir, transform=transform)
    validset = ImageFolder(root=valid_dir, transform=transform)
    testset = ImageFolder(root=test_dir, transform=transform)
    
    # Create DataLoaders
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(validset, batch_size=64, shuffle=True)
    test_loader = DataLoader(testset, batch_size=64, shuffle=True)
    
    # Save the images and labels for train, valid, and test sets
    save_img(train_loader, is_train=True, target_dir=target_dir)
    save_img(valid_loader, is_train=False, target_dir=target_dir)  # Validation set can be treated like a test set
    save_img(test_loader, is_train=False, target_dir=target_dir)
    
    print("CINIC-10 data processing done!")
    
## synthetic data
# Define a class for the arguments used in generate_synthetic_data
import numpy as np
from sklearn.model_selection import train_test_split
import os

def generate_synthetic_data(args, partition: dict, stats: dict, target_dir: str, test_size=0.2):
    """
    Generate synthetic data, split into train/test, and save them as CSV files in the target directory.

    Args:
        args: Object containing necessary parameters such as client_num, dimension, classes, etc.
        partition (dict): Dictionary to store partition information.
        stats (dict): Dictionary to store statistics for each client.
        target_dir (str): Directory to save the generated synthetic data.
        test_size (float): Fraction of the dataset to be used for testing.
    """
    def softmax(x):
        ex = np.exp(x)
        sum_ex = np.sum(np.exp(x))
        return ex / sum_ex

    class_num = 10 if args.classes <= 0 else args.classes
    samples_per_user = (
        np.random.lognormal(4, 2, args.client_num).astype(int) + 50
    ).tolist()

    w_global = np.zeros((args.dimension, class_num))
    b_global = np.zeros(class_num)

    mean_w = np.random.normal(0, args.gamma, args.client_num)
    mean_b = mean_w
    B = np.random.normal(0, args.beta, args.client_num)
    mean_x = np.zeros((args.client_num, args.dimension))

    diagonal = np.zeros(args.dimension)
    for j in range(args.dimension):
        diagonal[j] = np.power((j + 1), -1.2)
    cov_x = np.diag(diagonal)

    all_data = []
    all_targets = []
    data_cnt = 0

    for client_id in range(args.client_num):
        w = np.random.normal(mean_w[client_id], 1, (args.dimension, class_num))
        b = np.random.normal(mean_b[client_id], 1, class_num)

        if args.iid != 0:
            w = w_global
            b = b_global

        data = np.random.multivariate_normal(
            mean_x[client_id], cov_x, samples_per_user[client_id]
        )
        targets = np.zeros(samples_per_user[client_id], dtype=np.int32)

        for j in range(samples_per_user[client_id]):
            true_logit = np.dot(data[j], w) + b
            targets[j] = np.argmax(softmax(true_logit))

        all_data.append(data)
        all_targets.append(targets)

        partition["data_indices"][client_id] = list(
            range(data_cnt, data_cnt + len(data))
        )

        data_cnt += len(data)

        stats[client_id] = {}
        stats[client_id]["x"] = samples_per_user[client_id]
        stats[client_id]["y"] = Counter(targets.tolist())

    all_data = np.concatenate(all_data)
    all_targets = np.concatenate(all_targets)

    # Split the data into training and test sets
    train_data, test_data, train_targets, test_targets = train_test_split(
        all_data, all_targets, test_size=test_size, random_state=42
    )

    # Ensure the target directory exists
    train_dir = os.path.join(target_dir, 'train')
    test_dir = os.path.join(target_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Convert train and test data to Pandas DataFrame and save as CSV
    train_df = pd.DataFrame(train_data)
    train_df['label'] = train_targets
    test_df = pd.DataFrame(test_data)
    test_df['label'] = test_targets

    # Save as CSV files
    train_df.to_csv(os.path.join(train_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(test_dir, 'test.csv'), index=False)

    num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
    stats["samples_per_client"] = {
        "std": num_samples.mean().item(),
        "stddev": num_samples.std().item(),
    }

    print("Synthetic data generation, train/test split, and saving as CSV complete.")

# Now add functionality to process the data in train/test sets using DataLoader like in CIFAR-10
def process_synthetic_data(data_dir, target_dir):
    """
    Loads synthetic data, creates DataLoaders for train and test datasets, 
    and saves processed images.

    Args:
        data_dir (str): Directory where the synthetic data is stored.
        target_dir (str): Directory where processed data will be saved.
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load the synthetic data
    train_data = np.load(os.path.join(data_dir, 'train_data.npy'))
    train_targets = np.load(os.path.join(data_dir, 'train_targets.npy'))
    test_data = np.load(os.path.join(data_dir, 'test_data.npy'))
    test_targets = np.load(os.path.join(data_dir, 'test_targets.npy'))

    # Create synthetic datasets (similar to CIFAR-10)
    trainset = SyntheticDataset(train_data, train_targets, transform=transform)
    testset = SyntheticDataset(test_data, test_targets, transform=transform)

    # Create DataLoaders for training and testing
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    # Save images (this function needs to be implemented based on your requirements)
    save_img(train_loader, is_train=True, target_dir=target_dir)
    save_img(test_loader, is_train=False, target_dir=target_dir)

    print("Synthetic data processing and saving done!")

# Placeholder class to create a PyTorch Dataset for synthetic data
class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, target

# Define a class for the arguments used in generate_synthetic_data
class SyntheticDataArgs:
    def __init__(self):
        self.client_num = conf["num_parties"]  # Number of clients or parties
        self.dimension = conf["dimension"] if "dimension" in conf else 3072  # Dimension of the data, 32x32x3 = 3072 (like CIFAR-10)
        self.classes = conf["num_classes"]     # Number of classes
        self.gamma = conf.get("gamma", 1.0)    # Gamma, controls weight distribution, default is 1.0
        self.beta = conf.get("beta", 0.05)     # Beta, controls non-IID data distribution
        self.iid = conf.get("iid", True)       # IID setting, True if data should be IID

# Define a function to load synthetic data arguments from the conf file
def load_synthetic_data_args():
    return SyntheticDataArgs()


def download_data(data_dir, dataset_dir):
    
    #deleting dataset
    delete_data(data_dir)

    # Create the .data directory
    os.makedirs(data_dir, exist_ok=True)

    # Download and process the data based on the dataset_used in conf.py
    if conf["dataset_used"] == "mnist":
        process_MNIST(data_dir, dataset_dir)

    elif conf["dataset_used"] == "fmnist":
        process_FashionMNIST(data_dir, dataset_dir)

    elif conf["dataset_used"] == "cifar10":
        #process_CIFAR10(data_dir, dataset_dir)
        process_CIFAR10('/home/nkang90/FL-CL-code/cifar-10-python', '/home/nkang90/FL-CL-code/data/dataset')

    elif conf["dataset_used"] == "cifar100":
        #process_CIFAR100(data_dir, dataset_dir)
        process_CIFAR100('/home/nkang90/FL-CL-code/cifar-100-python', '/home/nkang90/FL-CL-code/data/dataset')

    elif conf["dataset_used"] == "cinic10":
        process_CINIC10('/home/nkang90/FL-bench-master/data/cinic10/raw', '/home/nkang90/FL-CL-code/data/dataset')

    elif conf["dataset_used"] == "usps":
        process_USPS(data_dir, dataset_dir)

    elif conf["dataset_used"] == "svhn":
        process_SVHN(data_dir, dataset_dir)

    elif conf["dataset_used"] == "kaggle":
        process_KAGGLE(dataset_dir)
        if conf["kaggle"]["dataset_used"] == "adult_income":
            adult_income_dataset()
            
    elif conf["dataset_used"] == "synthetic":
        # Example usage of synthetic data generation
        args = load_synthetic_data_args()  # Load arguments from conf.py
        partition = {"data_indices": {}}
        stats = {}
        generate_synthetic_data(args, partition, stats, target_dir=dataset_dir)

    else:
        print("Please check dataset_used in conf.py")
        
        

if __name__ == "__main__":
    download_data(data_dir='./data', dataset_dir='./data/dataset')



