import torch
from torchvision import models
from torch import nn
import torch.nn.functional as F
from conf import conf

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


class CNN_Model_old(nn.Module):

    def __init__(self):
        super(CNN_Model_old, self).__init__()
        self.cnn = nn.Sequential(
            ## Layer 1

            nn.Conv2d(3, 6, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            ## Layer 2
            nn.Conv2d(6, 16, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(

            ##Layer 3
            nn.Linear(256,120),#400
            nn.ReLU(),

            #Layer 4
            nn.Linear(120,84),
            nn.ReLU(),

            #Layer 5
            nn.Linear(84,84),
            nn.ReLU(),

            #Layer 6
            nn.Linear(84,256)
        )

        #Layer 7  classifier layer
        self.classifier = nn.Linear(256,10)

    def forward(self, input):
        x = self.cnn(input)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x, self.classifier(x)

class CNN_Model_old_1(nn.Module):
    def __init__(self):
        super(CNN_Model_old_1, self).__init__()
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

#28
class CNN_Model_28(nn.Module):
    def __init__(self):
        super(CNN_Model_28, self).__init__()
        
        # First Conv layer
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1) #28 28 128
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.3)
        
        # Second Conv layer
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # 14 14 128
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 14 14 256
        self.dropout2 = nn.Dropout(0.3)
        
        # Third, fourth, fifth convolution layer
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # 7 7 256
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # 7 7 512
        self.conv5 = nn.Conv2d(512, 256, kernel_size=3, padding=1)# 7 7 256
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 3 3 256
        self.dropout3 = nn.Dropout(0.3)

        # Fully Connected layers
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(256, 128)
        self.relu6 = nn.ReLU()
        self.dropout6 = nn.Dropout(0.5)
        
        self.classifier  = nn.Linear(128, 10)  # Output size is 10 for 10 classes

    def forward(self, x):
        # Convolutional layers
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.dropout1(x)
        
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.dropout2(x)
        
        x = self.pool3(self.relu3(self.conv5(self.conv4(self.conv3(x)))))
        x = self.dropout3(x)

        # Fully connected layers
        x = self.flatten(x)
        
        x = self.dropout4(self.relu4(self.fc1(x)))
        x = self.dropout5(self.relu5(self.fc2(x)))
        x = self.dropout6(self.relu6(self.fc3(x)))
        
        logits = self.classifier(x)
        
        return x, logits   # Return both logits and features

#32
class CNN_Model_32(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        
        # First Conv layer
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.3)
        
        # Second Conv layer
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.3)
        
        # Third, fourth, fifth convolution layer
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.3)

        # Fully Connected layers
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(256, 128)
        self.relu6 = nn.ReLU()
        self.dropout6 = nn.Dropout(0.5)
        
        self.classifier  = nn.Linear(128, 10)  # Output size is 10 for 10 classes

    def forward(self, x):
        # Convolutional layers
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.dropout1(x)
        
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.dropout2(x)
        
        x = self.pool3(self.relu3(self.conv5(self.conv4(self.conv3(x)))))
        x = self.dropout3(x)

        # Fully connected layers
        x = self.flatten(x)
        
        x = self.dropout4(self.relu4(self.fc1(x)))
        x = self.dropout5(self.relu5(self.fc2(x)))
        x = self.dropout6(self.relu6(self.fc3(x)))
        
        logits = self.classifier(x)
        
        return x, logits   # Return both logits and features

# as represents in FedLC papers 
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()

        # First Conv layer
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=0) #30 x 30 x 128
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) #15 x 15 x 128

        # Second Conv layer
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0) #13 x 13 x 128
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) #6 x 6 x 128

        # Third Conv layer
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0) #4 x 4 x 128
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 2 x 2 x 128

        # Fully Connected layers
        self.flatten = nn.Flatten()

        self.classifier = nn.Linear(2 * 2 * 128, 10) #512

    def forward(self, x):
        # Convolutional layers
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))

        # Fully connected layers
        x = self.flatten(x) # embeddings
        logits = self.classifier(x)

        return x, logits   # Return both logits and features



#v1
class ReTrainModelCNN(nn.Module):

    def __init__(self):
        super(ReTrainModelCNN, self).__init__()

        #Layer 7  classifier layer
        # self.classifier = nn.Linear(256,10) #256
        # self.classifier = nn.Linear(200,10)  # old 1
        # self.classifier = nn.Linear(128,10)
        self.classifier = nn.Linear(512,10) 

    def forward(self, input):

        return self.classifier(input)
    


class ReTrainModel(nn.Module):

    def __init__(self):
        super(ReTrainModel, self).__init__()

        #Layer 5  out layer
        self.out = nn.Linear(64,10) #256
        # check if kaggle dataset is used so select model on the base of dataset used!
        if conf['dataset_used'] == "kaggle":

            print("---------Re-Train Model for KAGGLE dataset selected!---------")

            if conf['kaggle']['dataset_used'] == "adult_income":
                self.out = nn.Linear(64,1) 

    def forward(self, input):

        return self.out(input)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal(m.weight, 1.0, 0.02)
        torch.nn.init.constant(m.bias, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal(m.weight, 0.0, 0.01)
        torch.nn.init.constant(m.bias, 0.0)




