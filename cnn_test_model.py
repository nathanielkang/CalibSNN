import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from sklearn.utils import shuffle
import numpy as np
from conf import conf
from utils import get_data
from fedavg.datasets import MyImageDataset

from torchvision.utils import make_grid
import matplotlib.pyplot as plt
train_datasets, val_datasets, test_dataset = get_data()



print("length train_datasets[0]",len(train_datasets[0]))
print("lenght val_datasets[0]",len(val_datasets[0]))
# test_dataset = test_dataset[: int(0.3 * train_data_portion)]
# print(f"test_Dataset lenght: {len(test_dataset)} > 30% of train_dataset")

train_data = MyImageDataset(train_datasets[0], conf['data_column'], conf['label_column'])
train_loader = DataLoader(train_data, batch_size=conf["batch_size"],shuffle=True)

val_data = MyImageDataset(val_datasets[0], conf['data_column'], conf['label_column'])
val_loader = DataLoader(val_data, batch_size=conf["batch_size"],shuffle=True)


class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.cnn = nn.Sequential(
            ## Layer 1
            nn.Conv2d(3, 6, 5, 1), #24 x 24 x 5 
            nn.ReLU(),
            nn.MaxPool2d(2), # 12×12×6

            ## Layer 2
            nn.Conv2d(6, 16, 5, 1),  #8×8×16
            nn.ReLU(),
            nn.MaxPool2d(2) # 4×4×16
        )

        self.fc = nn.Sequential(
            ## Layer 3
            nn.Linear(256, 120),
            nn.ReLU(),

            # Layer 4
            nn.Linear(120, 84),
            nn.ReLU(),

            # Layer 5
            # nn.Linear(84, 84),
            # nn.ReLU(),

            # Layer 6
            nn.Linear(84, 256),
            nn.ReLU()
        )

        # Layer 7 classifier layer
        self.classifier = nn.Linear(256, 10)

    def forward(self, input):
        x = self.cnn(input)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x, self.classifier(x)

# torch.nn.Conv2d(in_channels, out_channels, kernel_size,
# stride=1, padding=0, dilation=1, groups=1, bias=True,
#  padding_mode='zeros', device=None, dtype=None)

class CNN_Model_2(nn.Module):
    def __init__(self):
        super(CNN_Model_2, self).__init__()
        self.cnn = nn.Sequential(
            ## Layer 1
            nn.Conv2d(3, 16, 3, 1, padding= 1), # 28 x 28 x 16
            nn.ReLU(),
            nn.MaxPool2d(2), #14 × 14 × 16

            ## Layer 2
            nn.Conv2d(16, 32, 3, 1, padding=1),  # 14 x 14 x 32
            nn.ReLU(),
            nn.MaxPool2d(2), # 7 x 7 x 32

            ## Layer 3
            nn.Conv2d(32, 64, 3, 1, padding=1),  # 7 x 7 x 64
            nn.ReLU(),
            nn.MaxPool2d(2) # 3 x 3 x 64
        )

        self.fc = nn.Sequential(
            ## Layer 3
            nn.Linear(3*3*64, 500),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # Layer 7 classifier layer
        self.classifier = nn.Linear(500, 10)

    def forward(self, input):
        x = self.cnn(input)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x, self.classifier(x)

class CNN_Model_3(nn.Module):
    def __init__(self):
        super(CNN_Model_3, self).__init__()
        self.cnn = nn.Sequential(
            ## Layer 1
            nn.Conv2d(3, 32, 3, 1, padding=1),  # 28 x 28 x 32
            nn.ReLU(),
            nn.MaxPool2d(2), #  14 x 14 x 32
            ## Layer 2
            nn.Conv2d(32, 64, 3, 1, padding=1),  # 14 x 14 x 64
            nn.ReLU(),
            nn.MaxPool2d(2) # 7 x 7 x 64
        )

        self.fc = nn.Sequential(
            ## Layer 3
            nn.Linear(7*7*64, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            # Layer 4
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # Layer 7 classifier layer
        self.classifier = nn.Linear(256, 10)

    def forward(self, input):
        x = self.cnn(input)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x, self.classifier(x)


class CNN_Model_4(nn.Module):
    def __init__(self):
        super(CNN_Model_4, self).__init__()
        self.cnn = nn.Sequential(
            ## Layer 1
            nn.Conv2d(3, 32, 3, 1, padding=1),  # 28 x 28 x 32
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #  14 x 14 x 32
            
            ## Layer 2
            nn.Conv2d(32, 64, 3, 1, padding=1),  # 14 x 14 x 64
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 7 x 7 x 64

             ## Layer 3
            nn.Conv2d(64, 128, 3, 1, padding=1),  # 7 x 7 x 128
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # 3 x 3 x 128
        )

        self.fc = nn.Sequential(
            ## Layer 4
            nn.Linear(128 * 3 * 3, 200),
            nn.ReLU(),
 
        )

        # Layer 7 classifier layer
        self.classifier = nn.Linear(200, 10)

    def forward(self, input):
        x = self.cnn(input)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x, self.classifier(x)




model = CNN_Model_4()

def trainer_model(model, train_loader, val_loader, local_epochs, lr, momentum, weight_decay, device):
    """
    Train the provided model using the given training data loader.

    :param model: The PyTorch model to be trained
    :param train_loader: DataLoader for training data
    :param val_loader: DataLoader for validation data
    :param local_epochs: Number of local training epochs
    :param lr: Learning rate for the optimizer
    :param momentum: Momentum for SGD optimizer
    :param weight_decay: Weight decay for the optimizer
    :param device: Device to run the training on (e.g., 'cuda' or 'cpu')
    :return: Updated model state_dict 
    """

    # Move the model to the specified device
    model.to(device)

    # Define the optimizer and criterion
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()


    for epoch in range(local_epochs):
        model.train()
        total_l = 0
        total_dataset_size = 0
        correct_train = 0

        for batch_id, batch in enumerate(train_loader):
            """ I was appending total_dataset_size by lenght of batch that is always 2 (data,labels)
            """
            # total_dataset_size += len(batch)
            # print(total_dataset_size)

            data, target = batch

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            total_dataset_size += data.size()[0]

            optimizer.zero_grad()
            _, output = model(data)
            
            loss = criterion(output, target)
            # print(loss,type(loss))
            total_l += loss.item()

            loss.backward()

            optimizer.step()

            # Compute training accuracy
            _, predicted = torch.max(output.data, 1)
            correct_train += (predicted == target).sum().item()
        
        # Training accuracy and loss
        train_acc = 100.0 * (float(correct_train) / float(total_dataset_size))
        # train_acc = correct_train / total_dataset_size
        train_loss = total_l / total_dataset_size

        # Evaluate the model on the validation data
        model.eval()
        val_acc, val_loss = model_eval(model, val_loader, criterion, device)

        print("Epoch {0} done. Train Loss: {1:.4f}, Train Acc: {2:.2f}%, Eval Loss: {3:.4f}, Eval Acc: {4:.2f}%".format(epoch, train_loss, train_acc, val_loss, val_acc))
        # print("Epoch {0} done. Train Loss: {1}, Eval Loss: {2}, Eval Acc: {3}".format(epoch, total_l / total_dataset_size, val_loss, val_acc))
    
    # test_data = MyImageDataset(test_dataset, conf['data_column'], conf['label_column'])
    # test_loader = DataLoader(test_data, batch_size=conf["batch_size"],shuffle=True)
    # model.eval()
    # test_acc, test_loss = model_eval(model, test_loader, criterion, device)
    # print(f"Test Accuracy: {test_acc}%,Test Loss: {test_loss}%")
    

    return model.state_dict()

@torch.no_grad()
def model_eval(model, data_loader, criterion, device):
    """
    Evaluate the model on the provided data loader.

    :param model: The PyTorch model to be evaluated
    :param data_loader: DataLoader for evaluation data
    :param criterion: Loss criterion
    :param device: Device to run the evaluation on (e.g., 'cuda' or 'cpu')
    :return: Accuracy and average loss on the evaluation data
    """


    total_loss = 0.0
    correct = 0
    dataset_size = 0    
    
    for batch_id, batch in enumerate(data_loader):
        data, target = batch
        dataset_size += data.size()[0]
        
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        _, output = model(data)

        total_loss += criterion(output, target) # sum up batch loss
        # pred = output.data.max(1)[1]  # get the index of the max log-probability
        max_prob, pred = torch.max(output.data, 1)

        
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(dataset_size))
    total_l = total_loss.cpu().detach().numpy() / dataset_size

    return acc, total_l


model_state = trainer_model(model = model,
              train_loader = train_loader,
              val_loader = val_loader,
              local_epochs = 15,
              lr = 0.005,
              momentum = 0.9,
              weight_decay = 1e-5,
              device = "cpu")

