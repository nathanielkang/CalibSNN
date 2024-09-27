import torch
import numpy as np
import pandas as pd
from fedavg.datasets import get_dataset
from conf import conf
import os
from loss_function.select_loss_fn import selected_loss_function

class Client(object):

    def __init__(self, conf, model, train_df, val_df):
        """
        :param conf: configuration
        :param model: model 
        :param train_dataset: Train Dataset
        :param val_dataset: Val Dataset
        """

        self.conf = conf

        self.local_model = model
        self.train_df = train_df
        self.train_dataset = get_dataset(conf, self.train_df, conf['train_load_data'])
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"],shuffle=True)

        self.val_df = val_df
        self.val_dataset = get_dataset(conf, self.val_df, conf['eval_load_data'])
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=conf["batch_size"],shuffle=True)

    def local_train(self, model, client_id, golbal_epochs):

        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        if self.conf["client_optimizer"] == "SGD":
            optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'],weight_decay=self.conf["weight_decay"])
        
        elif self.conf["client_optimizer"] == "Adam":
            optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.conf['lr'],weight_decay=self.conf["weight_decay"])
        
        else:
            print("Please select client_optimizer in conf.py!")
        
        
        criterion = selected_loss_function(loss=self.conf['train_loss_criterion'])

        # Initialize lists to store training information
        local_training_info = []

        
        for e in range(self.conf["local_epochs"]):
            self.local_model.train()
            total_l = 0
            total_dataset_size = 0
            for batch_id, batch in enumerate(self.train_loader):
                if self.conf['train_contrastive_learning']:
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                    img1, img2, label1, label2 = batch
                    img1, img2 = img1.to(device), img2.to(device)
                    label1, label2 = label1.to(device), label2.to(device)

                    optimizer.zero_grad()

                    # Forward pass
                    embeddings1, logits1 = self.local_model(img1)
                    embeddings2, logits2  = self.local_model(img2)

                    # Compute loss
                    loss = criterion(logits1, label1, embeddings1, logits2, label2, embeddings2)
                    total_l += loss.item()

                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()

                    total_dataset_size += label1.size(0) + label2.size(0)
                else:
                    # total_dataset_size += len(batch)
                    data, target = batch
                    if torch.cuda.is_available():
                        data = data.cuda()
                        target = target.cuda()#for tabular it gives [64,]
                    
                    
                
                    # if dataset is tabular then convert the target to 2d array [64,1]
                    if self.conf['data_type'] == 'tabular':
                        target = target.float().view(-1, 1)
                    


                    total_dataset_size += data.size()[0]

                    optimizer.zero_grad()
                    feature, output = self.local_model(data)

                    loss = criterion(output, target)
                    total_l += loss.item()
                    
                    loss.backward()
                    optimizer.step()

            acc, eval_loss = self.model_eval() # eval loss and acc
            train_loss = total_l / total_dataset_size
            # Append training information for the current epoch
            local_training_info.append({
                'golbal_epoch':golbal_epochs,
                'client_id': client_id,
                'epoch': e,
                'train_loss': train_loss,#loss.item()
                'eval_loss': eval_loss,
                'eval_acc': acc,
                'global_acc': None,
                'global_loss': None
            })

            
            print("Epoch {0} done. train_loss ={1}, eval_loss = {2}, eval_acc={3}".format(e, train_loss, eval_loss, acc))

            # print("Epoch {0} done. train_loss ={1}, eval_loss = {2}, eval_acc={3}".format(e, total_l / total_dataset_size, eval_loss, acc))
       
        return self.local_model.state_dict(),local_training_info

    @torch.no_grad()
    def model_eval(self):
        self.local_model.eval()

        total_val_loss = 0.0
        total_correct = 0
        total_samples = 0

       

        criterion = selected_loss_function(loss=self.conf['eval_loss_criterion'])
        
        for batch_id, batch in enumerate(self.val_loader):
            if conf['eval_contrastive_learning']:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                img1, img2, label1, label2 = batch
                img1, img2 = img1.to(device), img2.to(device)
                label1, label2 = label1.to(device), label2.to(device)

                # Forward pass
                embeddings1, logits1 = self.local_model(img1)
                embeddings2, logits2 = self.local_model(img2)

                # Compute loss
                loss = criterion(logits1, label1, embeddings1, logits2, label2, embeddings2)
                total_val_loss += loss.item()

                # Compute accuracy
                _, predicted1 = torch.max(logits1, 1)
                _, predicted2 = torch.max(logits2, 1)
                correct1 = (predicted1 == label1).sum().item()
                correct2 = (predicted2 == label2).sum().item()
                total_correct += correct1 + correct2
                total_samples += label1.size(0) + label2.size(0)
          
            else:
                data, target = batch
                
                

                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()#for tabular it gives [64,]
                # if dataset is tabular then convert the target to 2d array [64,1]
                if self.conf['data_type'] == 'tabular':
                    target = target.float().view(-1, 1)
                    
                
                _, output = self.local_model(data)
                
                
                total_val_loss += criterion(output, target)    # sum up batch loss

                if self.conf['classification_type']=="multi":
                    pred = output.data.max(1)[1]
                elif self.conf['classification_type']=="binary":
                    pred = (torch.sigmoid(output) > 0.5).float()
                else:
                    print("Please check type of classfication!")
                
                total_correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                total_samples += data.size()[0]
        if self.conf['eval_contrastive_learning']:     
            accuracy = 100.0 * total_correct / total_samples
            avg_val_loss = total_val_loss / len(self.val_loader)
            
            
        else:    
            accuracy = 100.0 * (float(total_correct) / float(total_samples))
            avg_val_loss = total_val_loss.cpu().detach().numpy() / total_samples

        return accuracy, avg_val_loss

    def _cal_mean_cov(self,features):
        """
        :param features: output features，(batch_size, 256)
        :return:
        """
        features = np.array(features)
        mean = np.mean(features, axis=0)
        cov = np.cov(features.T, bias=1)
        return mean,cov

    def cal_distributions(self, model,load_pair_data =False):
        """
        :param feature:
        :return:
        """
        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        self.local_model.eval()

        features = []
        mean = []
        cov = []
        length = []

        for i in range(self.conf["num_classes"]):
            train_i = self.train_df[self.train_df[self.conf['label_column']] == i]
            train_i_dataset = get_dataset(self.conf, train_i, load_pair_data)
            
            if len(train_i_dataset) > 0:
                train_i_loader = torch.utils.data.DataLoader(train_i_dataset, batch_size=self.conf["batch_size"],
                                                             shuffle=True)
                for batch_id, batch in enumerate(train_i_loader):
                    data, target = batch

                    if torch.cuda.is_available():
                        data = data.cuda()

                    feature, _ = self.local_model(data)
                    
                    features.extend(feature.tolist())

                f_mean, f_cov = self._cal_mean_cov(features)
                

            else:
                if conf['model_name'] == "mlp":
                    f_mean = np.zeros((64,)) #256
                    f_cov = np.zeros((64,64))#256

                elif conf['model_name'] == "cnn":
                    # f_mean = np.zeros((256,)) #256
                    # f_cov = np.zeros((256,256))#256
                    
                    # f_mean = np.zeros((200,))  # old 1
                    # f_cov = np.zeros((200,200))

                    # f_mean = np.zeros((128,)) # 32 and 28 model
                    # f_cov = np.zeros((128,128))

                    f_mean = np.zeros((512,)) 
                    f_cov = np.zeros((512,512))
                    
                    

            mean.append(f_mean)
            cov.append(f_cov)
            length.append(len(train_i))

        return mean, cov, length







