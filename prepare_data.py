import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import pandas as pd
from conf import conf
import argparse


def save_dataset_images(loader, is_train, target_dir, dataset_name):
    """Save images from dataloader to directory structure."""
    if is_train:
        target_dir = os.path.join(target_dir, 'train')
        index_file = os.path.join(target_dir, 'train.csv')
    else:
        target_dir = os.path.join(target_dir, 'test')
        index_file = os.path.join(target_dir, 'test.csv')

    os.makedirs(target_dir, exist_ok=True)

    num = 0
    index_fname = []
    index_label = []

    print(f"Saving {dataset_name} {'train' if is_train else 'test'} images...")
    
    for _, batch_data in enumerate(loader):
        data, label = batch_data
        for d, l in zip(data, label):
            # Generate pic and save it into the directory
            result_dir = os.path.join(target_dir, str(l.item()))
            if not os.path.exists(result_dir):
                os.makedirs(result_dir, exist_ok=True)

            # Create image/file names
            file = os.path.join(result_dir, f"{l.item()}-{num}.png")
            index_fname.append(file)
            index_label.append(l.item())

            # Save images
            save_image(d.data, file)
            num += 1

    # Save index
    index = pd.DataFrame({
        conf["data_column"]: index_fname,
        conf["label_column"]: index_label
    })
    index.to_csv(index_file, index=False)
    print(f"Saved {num} images to {target_dir}")


def prepare_mnist(data_dir='./data', target_dir='./data/dataset'):
    """Prepare MNIST dataset."""
    print("Preparing MNIST dataset...")
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    
    trainset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    
    save_dataset_images(train_loader, is_train=True, target_dir=target_dir, dataset_name="MNIST")
    save_dataset_images(test_loader, is_train=False, target_dir=target_dir, dataset_name="MNIST")
    print("MNIST data preparation done!")


def prepare_cifar10(data_dir='./data', target_dir='./data/dataset'):
    """Prepare CIFAR-10 dataset."""
    print("Preparing CIFAR-10 dataset...")
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    
    save_dataset_images(train_loader, is_train=True, target_dir=target_dir, dataset_name="CIFAR-10")
    save_dataset_images(test_loader, is_train=False, target_dir=target_dir, dataset_name="CIFAR-10")
    print("CIFAR-10 data preparation done!")


def prepare_usps(data_dir='./data', target_dir='./data/dataset'):
    """Prepare USPS dataset."""
    print("Preparing USPS dataset...")
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    
    trainset = torchvision.datasets.USPS(
        root=data_dir, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.USPS(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    
    save_dataset_images(train_loader, is_train=True, target_dir=target_dir, dataset_name="USPS")
    save_dataset_images(test_loader, is_train=False, target_dir=target_dir, dataset_name="USPS")
    print("USPS data preparation done!")


def main():
    parser = argparse.ArgumentParser(description='Prepare datasets for CalibSNN')
    parser.add_argument('--datasets', nargs='+', 
                       default=['mnist', 'cifar10', 'usps'],
                       choices=['mnist', 'cifar10', 'usps'],
                       help='Datasets to prepare')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory to download raw data')
    parser.add_argument('--target_dir', type=str, default='./data/dataset',
                       help='Directory to save processed data')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.target_dir, exist_ok=True)
    
    # Prepare selected datasets
    for dataset in args.datasets:
        if dataset == 'mnist':
            prepare_mnist(args.data_dir, args.target_dir)
        elif dataset == 'cifar10':
            prepare_cifar10(args.data_dir, args.target_dir)
        elif dataset == 'usps':
            prepare_usps(args.data_dir, args.target_dir)
    
    print(f"\nAll datasets prepared successfully!")
    print(f"Data saved to: {args.target_dir}")


if __name__ == "__main__":
    main() 