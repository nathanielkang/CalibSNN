import torch
import numpy as np
import pandas as pd
import time
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
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"], shuffle=True)

        self.val_df = val_df
        self.val_dataset = get_dataset(conf, self.val_df, conf['eval_load_data'])
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=conf["batch_size"], shuffle=True)

    def local_train(self, model, client_id, global_epochs):

        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        # Optimizer setup
        if self.conf["client_optimizer"] == "SGD":
            optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'],
                                        weight_decay=self.conf["weight_decay"])
        elif self.conf["client_optimizer"] == "Adam":
            optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.conf['lr'], weight_decay=self.conf["weight_decay"])
        else:
            raise ValueError("Please select client_optimizer in conf.py!")

        criterion = selected_loss_function(loss=self.conf['train_loss_criterion'])
        local_training_info = []

        # Training loop with manual timeout check
        for e in range(self.conf["local_epochs"]):
            start_time = time.time()  # Start tracking time for the epoch

            total_loss, total_dataset_size = 0, 0
            try:
                # Run the training for this epoch
                self.local_model.train()
                for batch_id, batch in enumerate(self.train_loader):
                    # Check if we've exceeded the 30-second timeout
                    if time.time() - start_time > 30:
                        raise TimeoutError(f"Timeout: Client {client_id} took too long in epoch {e}. Skipping.")

                    if self.conf['train_contrastive_learning']:
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        img1, img2, label1, label2 = batch
                        img1, img2 = img1.to(device), img2.to(device)
                        label1, label2 = label1.to(device), label2.to(device)

                        optimizer.zero_grad()

                        embeddings1, logits1 = self.local_model(img1)
                        embeddings2, logits2 = self.local_model(img2)

                        loss = criterion(logits1, label1, embeddings1, logits2, label2, embeddings2)
                        total_loss += loss.item()
                        loss.backward()
                        optimizer.step()

                        total_dataset_size += label1.size(0) + label2.size(0)

                    else:
                        data, target = batch
                        if torch.cuda.is_available():
                            data = data.cuda()
                            target = target.cuda()

                        if self.conf['data_type'] == 'tabular':
                            target = target.float().view(-1, 1)

                        total_dataset_size += data.size()[0]

                        optimizer.zero_grad()
                        _, output = self.local_model(data)

                        loss = criterion(output, target)
                        total_loss += loss.item()
                        loss.backward()
                        optimizer.step()

                # Log training information after each epoch
                acc, eval_loss = self.model_eval()
                train_loss = total_loss / total_dataset_size
                local_training_info.append({
                    'global_epoch': global_epochs,
                    'client_id': client_id,
                    'epoch': e,
                    'train_loss': train_loss,
                    'eval_loss': eval_loss,
                    'eval_acc': acc,
                    'global_acc': None,
                    'global_loss': None
                })
                print(f"Epoch {e} done. train_loss = {train_loss}, eval_loss = {eval_loss}, eval_acc = {acc}")

            except TimeoutError as te:
                print(te)  # Log the timeout error
                break  # Skip this client if training takes too long

        return self.local_model.state_dict(), local_training_info
    
    @torch.no_grad()
    def model_eval(self):
        """
        Evaluation logic.
        """
        self.local_model.eval()
        total_val_loss = 0.0
        total_correct = 0
        total_samples = 0
        criterion = selected_loss_function(loss=self.conf['eval_loss_criterion'])
        
        start_time = time.time()  # Start tracking time for evaluation

        for batch_id, batch in enumerate(self.val_loader):
            
            if time.time() - start_time > 30:  # Timeout after 30 seconds
                raise TimeoutError(f"Timeout: Evaluation took too long for batch {batch_id}. Skipping.")
            
            if conf['eval_contrastive_learning']:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                img1, img2, label1, label2 = batch
                img1, img2 = img1.to(device), img2.to(device)
                label1, label2 = label1.to(device), label2.to(device)

                embeddings1, logits1 = self.local_model(img1)
                embeddings2, logits2 = self.local_model(img2)

                loss = criterion(logits1, label1, embeddings1, logits2, label2, embeddings2)
                total_val_loss += loss.item()

                if self.conf['classification_type'] == "multi":
                    _, predicted1 = torch.max(logits1, 1)
                    _, predicted2 = torch.max(logits2, 1)
                elif self.conf['classification_type'] == "binary":
                    predicted1 = (torch.sigmoid(logits1) > 0.5).float().squeeze()
                    predicted2 = (torch.sigmoid(logits2) > 0.5).float().squeeze()
                else:
                    raise ValueError("Please check type of classification! (multi or binary)")

                correct1 = (predicted1 == label1).sum().item()
                correct2 = (predicted2 == label2).sum().item()
                total_correct += correct1 + correct2
                total_samples += label1.size(0) + label2.size(0)

            else:
                data, target = batch
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                if self.conf['data_type'] == 'tabular':
                    target = target.float().view(-1, 1)

                _, output = self.local_model(data)
                total_val_loss += criterion(output, target).item()

                if self.conf['classification_type'] == "multi":
                    pred = output.data.max(1)[1]
                elif self.conf['classification_type'] == "binary":
                    pred = (torch.sigmoid(output) > 0.5).float()
                else:
                    raise ValueError("Please check type of classification! (multi or binary)")

                total_correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                total_samples += data.size()[0]

        accuracy = 100.0 * (float(total_correct) / float(total_samples))
        avg_val_loss = total_val_loss / total_samples
        return accuracy, avg_val_loss

    def _cal_mean_cov(self, features):
        """
        Calculate the mean and covariance of the provided features.
        :param features: output features, (batch_size, feature_size)
        :return: mean and covariance of the features
        """
        features = np.array(features)
        mean = np.mean(features, axis=0)
        cov = np.cov(features.T, bias=1)
        return mean, cov

    def cal_distributions(self, model, load_pair_data=False):
        """
        Calculate the distributions (mean, covariance, and length) of features for each class.
        :param model: model to evaluate
        :param load_pair_data: whether to load paired data (for contrastive learning)
        :return: means, covariances, and lengths of the features for each class
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
                train_i_loader = torch.utils.data.DataLoader(train_i_dataset, batch_size=self.conf["batch_size"], shuffle=True)
                for batch_id, batch in enumerate(train_i_loader):
                    data, target = batch

                    if torch.cuda.is_available():
                        data = data.cuda()

                    feature, _ = self.local_model(data)
                    features.extend(feature.tolist())

                f_mean, f_cov = self._cal_mean_cov(features)

            else:
                # Handle different model architectures with different feature sizes
                if self.conf['model_name'] == "mlp":
                    f_mean = np.zeros((64,))  # 64 is a typical feature size for MLP
                    f_cov = np.zeros((64, 64))
                elif self.conf['model_name'] == "cnn":
                    f_mean = np.zeros((512,))  # CNNs often have larger feature sizes
                    f_cov = np.zeros((512, 512))
                else:
                    raise ValueError("Unknown model type. Check your model configuration!")

            mean.append(f_mean)
            cov.append(f_cov)
            length.append(len(train_i))

        return mean, cov, length
