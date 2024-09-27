import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random
class MyTabularDataset(Dataset):

    def __init__(self, dataset, label_col):
        """
        :param dataset: dataset, DataFrame
        :param label_col: name of your column
        """

        self.label = torch.LongTensor(dataset[label_col].values)

        self.data = dataset.drop(columns=[label_col]).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        label = self.label[index]
        data = self.data[index]

        return torch.tensor(data).float(), label

class MyImageDataset(Dataset):
    def __init__(self, dataset, file_col, label_col):
        """
        :param dataset: dataset, DataFrame
        :param file_col:  file name of the column
        :param label_col:  label name of the column
        """
        self.file = dataset[file_col].values
        self.label = dataset[label_col].values

        self.normalize = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):

        label = torch.tensor(self.label[index])

        data = Image.open(self.file[index])
        data = self.normalize(data)


        return data, label

class VRDataset(Dataset):
    def __init__(self, data, label):
        """
        :param dataset: dataset, DataFrame
        :param file_col:  file name of the column
        :param label_col:  label name of the column
        """
        self.data = data

        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        data = torch.tensor(self.data[index])
        label = torch.tensor(self.label[index])

        return data, label

class PairsImageDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        img1, label1 = self.dataset[index]
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                index2 = random.randint(0, len(self.dataset) - 1)
                img2, label2 = self.dataset[index2]
                if label2 == label1:
                    break
        else:
            while True:
                index2 = random.randint(0, len(self.dataset) - 1)
                img2, label2 = self.dataset[index2]
                if label2 != label1:
                    break
        return img1, img2, label1, label2

    def __len__(self):
        return len(self.dataset)

def get_dataset(conf, data, load_pair_data):
    """
    :param conf: Configuration
    :param data: Data (DataFrame)
    :return:
    """
    if conf['data_type'] == 'tabular':
        dataset = MyTabularDataset(data, conf['label_column'])
    elif conf['data_type'] == 'image':
        
        
        dataset = MyImageDataset(data, conf['data_column'], conf['label_column'])
        if load_pair_data:
            dataset = PairsImageDataset(dataset)
    else:
        return None
    return dataset






