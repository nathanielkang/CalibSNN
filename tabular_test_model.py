from utils import get_data
from fedavg.datasets import MyTabularDataset
from conf import conf

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class MLP(nn.Module):

    def __init__(self, n_feature, n_hidden, n_output, dropout=0.5):
        
        super(MLP, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)

        self.hidden_1 = torch.nn.Linear(n_feature, n_hidden)  # 1 hidden layer
        self.bn1 = torch.nn.BatchNorm1d(n_hidden)

        self.hidden_2 = torch.nn.Linear(n_hidden, n_hidden // 2) # 2 hiddenlayer
        self.bn2 = torch.nn.BatchNorm1d(n_hidden // 2)

        self.hidden_3 = torch.nn.Linear(n_hidden // 2, n_hidden // 4)  # 3 hidden layer
        self.bn3 = torch.nn.BatchNorm1d(n_hidden // 4)

        self.hidden_4 = torch.nn.Linear(n_hidden // 4, n_hidden // 8)  # 4 hidden layer
        self.bn4 = torch.nn.BatchNorm1d(n_hidden // 8)

        self.out = torch.nn.Linear(n_hidden // 8, n_output)  # 5 output layer

    def forward(self, X):
        x = X.view(X.shape[0], -1)

        x = F.relu(self.hidden_1(x))  # hidden layer 1
        x = self.dropout(self.bn1(x))

        x = F.relu(self.hidden_2(x))  # hidden layer 2
        x = self.dropout(self.bn2(x))

        x = F.relu(self.hidden_3(x))  # hidden layer 3
        x = self.dropout(self.bn3(x))

        x = F.relu(self.hidden_4(x))  # hidden layer 4
        feature = self.dropout(self.bn4(x))

        x = self.out(feature)

        return feature, x


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
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    if conf["client_optimizer"] == "SGD":
        print("optimizer used: ", "SGD")
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
             
    elif conf["client_optimizer"] == "Adam":
        print("optimizer used: ", "Adam")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    else:
        print("Please select client_optimizer in conf.py!")
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()


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
            # print(torch.unique(target))

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            if conf['data_type'] == 'tabular':
                target = target.float().view(-1, 1)
            # print("total_dataset_size", data.size()[0])
            total_dataset_size += data.size()[0]

            optimizer.zero_grad()
            _, output = model(data)
            # print("output",output)
            # print("taget", target.float().view(-1, 1))
            loss = criterion(output, target)

            # print(loss,type(loss))
            total_l += loss.item()

            loss.backward()

            optimizer.step()

            # Compute training accuracy
            # _, predicted = torch.max(output.data, 1)
            predicted = (torch.sigmoid(output) > 0.5).float()
            # print(predicted)
            correct_train += predicted.eq(target.data.view_as(predicted )).cpu().sum().item()
            # print("correct_train",predicted .eq(target.data.view_as(predicted )).cpu().sum().item())
        # Training accuracy and loss
        train_acc = 100.0 * (float(correct_train) / float(total_dataset_size))
        # train_acc = correct_train / total_dataset_size
        train_loss = total_l / total_dataset_size

        # Evaluate the model on the validation data
        model.eval()
        val_acc, val_loss = model_eval(model, val_loader, criterion, device)

        print("Epoch {0} done. Train Loss: {1:.4f}, Train Acc: {2:.2f}%, Eval Loss: {3:.4f}, Eval Acc: {4:.2f}%".format(epoch, train_loss, train_acc, val_loss, val_acc))

    

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
        target = target.float().view(-1, 1)
        _, output = model(data)
        

        total_loss += criterion(output, target) # sum up batch loss
        # pred = output.data.max(1)[1]  # get the index of the max log-probability
        # max_prob, pred = torch.max(output.data, 1)
        pred = (torch.sigmoid(output) > 0.5).float()
        
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(dataset_size))
    total_l = total_loss.cpu().detach().numpy() / dataset_size

    return acc, total_l


if __name__ == "__main__":
    
    train_datasets, val_datasets, test_dataset = get_data()
    model = MLP(n_feature = 14, n_hidden = 512, n_output = 1, dropout=0.5)

    for i in range(conf['num_parties']):
        train_data = MyTabularDataset(train_datasets[i],conf['label_column'])
        train_loader = DataLoader(train_data, batch_size=conf["batch_size"],shuffle=True, drop_last=True)

        val_data = MyTabularDataset(val_datasets[i],conf['label_column'])
        val_loader = DataLoader(val_data, batch_size=conf["batch_size"],shuffle=True, drop_last=True)

        model_state = trainer_model(model = model,
                        train_loader = train_loader,
                        val_loader = val_loader,
                        local_epochs = 15,
                        lr = 0.005,
                        momentum = 0.9,
                        weight_decay = 1e-5,
                        device = "cpu")
        print(f"\n\n-------client_{i} done-------\n\n")




