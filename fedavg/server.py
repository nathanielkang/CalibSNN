import torch
from fedavg.datasets import get_dataset, VRDataset
from conf import conf
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc  # added ------
from sklearn.preprocessing import label_binarize  # added ------
from loss_function.select_loss_fn import selected_loss_function

class Server(object):

    def __init__(self, conf, model, test_df):

        self.conf = conf

        self.global_model = model

        self.test_dataset = get_dataset(conf, test_df, conf['test_load_data'])
        
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=conf["batch_size"], shuffle=False)

    def model_aggregate(self, clients_model, weights):

        new_model = {}

        for name, params in self.global_model.state_dict().items():
            new_model[name] = torch.zeros_like(params)

        for key in clients_model.keys():

            for name, param in clients_model[key].items():
                new_model[name] = new_model[name] + \
                    clients_model[key][name] * weights[key]

        self.global_model.load_state_dict(new_model)

    @torch.no_grad()
    def model_eval(self):
        self.global_model.eval()

        total_test_loss = 0.0
        total_correct = 0
        total_samples = 0

        predict_prob = []
        labels = []

        criterion = selected_loss_function(loss=self.conf['test_loss_criterion'])


        for batch_id, batch in enumerate(self.test_loader):
            if self.conf['test_contrastive_learning']:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                img1, img2, label1, label2 = batch
                img1, img2 = img1.to(device), img2.to(device)
                label1, label2 = label1.to(device), label2.to(device)

                # Forward pass
                embeddings1, logits1 = self.global_model(img1)
                embeddings2, logits2 = self.global_model(img2)

                # Compute loss
                loss = criterion(logits1, label1, embeddings1, logits2, label2, embeddings2)
                total_test_loss += loss.item()

                # Compute accuracy
                if self.conf['classification_type']=="multi":
                    _, predicted1 = torch.max(logits1, 1)
                    _, predicted2 = torch.max(logits2, 1)
                elif self.conf['classification_type']=="binary":
                    predicted1 = (torch.sigmoid(logits1) > 0.5).float().squeeze()
                    predicted2 = (torch.sigmoid(logits2) > 0.5).float().squeeze()
                else:
                    raise ValueError("Please check type of classfication! (multi or binary)")
                    
                correct1 = (predicted1 == label1).sum().item()
                correct2 = (predicted2 == label2).sum().item()
                total_correct += correct1 + correct2
                total_samples += label1.size(0) + label2.size(0)
            else:
                data, target = batch
                

                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda() #for tabular it gives [64,]
                
                
                
                # if dataset is tabular then convert the target to 2d array [64,1]
                if self.conf['data_type'] == 'tabular':
                    target = target.float().view(-1, 1)


                _, output = self.global_model(data)
                # print("output gloabl",output)

                total_test_loss += criterion(output, target)  # sum up batch loss
                # get the index of the max log-probability
                # pred = output.data.max(1)[1]
                # print("torch.sigmoid(output)",torch.sigmoid(output))
                ## pred = (torch.sigmoid(output) > 0.5).float()
                # print("pred gloabl",pred)
                if self.conf['classification_type']=="multi":
                    pred = output.data.max(1)[1]
                elif self.conf['classification_type']=="binary":
                    pred = (torch.sigmoid(output) > 0.5).float()
                else:
                    raise ValueError("Please check type of classication! (binary or multi)")

                # predict_prob.extend(output.data[:, 1].tolist())
                labels.extend(target.data.cpu().tolist())
                total_correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                total_samples += data.size()[0]
                # print("target.data.view_as(pred)",target.data.view_as(pred))
                # print("pred.eq(target.data.view_as(pred)).cpu().sum().item()",pred.eq(target.data.view_as(pred)).cpu().sum().item())
        
        if self.conf['test_contrastive_learning']:
            accuracy = 100.0 * total_correct / total_samples
            avg_test_loss = total_test_loss / len(self.test_loader)  
        else:
            accuracy = 100.0 * (float(total_correct) / float(total_samples))
            avg_test_loss = total_test_loss.cpu().detach().numpy() / total_samples
            # print("roc_auc = {}".format(roc_auc_score(labels,predict_prob)))

        return accuracy, avg_test_loss

    @torch.no_grad()
    def model_eval_vr(self, eval_vr, label):
        """
        :param eval_vr:
        :param label:
        :return: Test re-trained model
        """

        self.retrain_model.eval()

        eval_dataset = VRDataset(eval_vr, label)
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)

        total_loss = 0.0
        correct = 0
        dataset_size = 0

        criterion = selected_loss_function(loss=self.conf['retrain_loss_criterion'])

        # criterion = torch.nn.CrossEntropyLoss()

        for batch_id, batch in enumerate(eval_loader):
            data, target = batch
            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()#for tabular it gives [64,]
            
            # if dataset is tabular then convert the target to 2d array [64,1]
            if conf['data_type'] == 'tabular':
                target = target.float().view(-1, 1)

            

            output = self.retrain_model(data)

            total_loss += criterion(output, target)  # sum up batch loss
            # get the index of the max log-probability
            if self.conf['classification_type']=="multi":
                pred = output.data.max(1)[1]
            elif self.conf['classification_type']=="binary":
                pred = (torch.sigmoid(output) > 0.5).float()
            else:
                raise ValueError("Please check type of classication! (binary or multi)")

            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss.cpu().detach().numpy() / dataset_size
        return acc, total_l

    def retrain_vr(self, vr, label, eval_vr, classifier):
        """
        :param vr:
        :param label:
        :return: Return re-trained model

        """
        re_training_info = []
        self.retrain_model = classifier
        retrain_dataset = VRDataset(vr, label)

        retrain_loader = torch.utils.data.DataLoader(
            retrain_dataset, batch_size=self.conf["batch_size"], shuffle=True)

        if self.conf['re_train_optimizer'] == "SGD":
            optimizer = torch.optim.SGD(self.retrain_model.parameters(
            ), lr=self.conf['retrain']['lr'], momentum=self.conf['momentum'], weight_decay=self.conf["weight_decay"])

        elif self.conf['re_train_optimizer'] == "Adam":
            optimizer = torch.optim.Adam(
                self.retrain_model.parameters(), lr=self.conf['retrain']['lr'], weight_decay=self.conf['retrain']["weight_decay"])

        else:
            raise ValueError("Please select re_train_optimizer in conf.py!")

        # criterion = torch.nn.CrossEntropyLoss()
        criterion = selected_loss_function(loss=self.conf['retrain_loss_criterion'])

        total_dataset_size = 0
        total_dataset_loss = 0

        for e in range(self.conf["retrain"]["epoch"]):

            self.retrain_model.train()

            for batch_id, batch in enumerate(retrain_loader):
                data, target = batch
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()#for tabular it gives [64,]
                
                # if dataset is tabular then convert the target to 2d array [64,1]
                if self.conf['data_type'] == 'tabular':
                    target = target.float().view(-1, 1)
                
                total_dataset_size += data.size()[0]    

                optimizer.zero_grad()
                output = self.retrain_model(data)

                loss = criterion(output, target)
                total_dataset_loss += loss
                loss.backward()

                optimizer.step()
            
            training_loss = total_dataset_loss/total_dataset_size

            acc, eval_loss = self.model_eval_vr(eval_vr, label)
            re_training_info.append({
                'retraining_epoch': e,
                'train_loss': training_loss.item(),
                'eval_loss': eval_loss,
                'eval_acc': acc,
                'retraing_global_acc': None,
                'retraing_global_loss': None
            })
            print("Retraining epoch {0} done. train_loss ={1}, eval_loss = {2}, eval_acc={3}".format(
                e, training_loss, eval_loss, acc))

        return self.retrain_model, re_training_info

    def cal_global_gd(self, client_mean, client_cov, client_length):
        """
        :param client_mean: Participanting clients' mean, dictionary
        :param client_cov:  Participanting clients' covariance matrix, dictionary
        :param client_length: Participanting clients' dataset size, dictionary
        :return:
        """

        g_mean = []
        g_cov = []

        clients = list(client_mean.keys())

        for c in range(len(client_mean[clients[0]])):

            mean_c = np.zeros_like(client_mean[clients[0]][0])
            n_c = 0
            # Dataset size for each class
            for k in clients:
                n_c += client_length[k][c]

            cov_ck = np.zeros_like(client_cov[clients[0]][0])
            mul_mean = np.zeros_like(client_cov[clients[0]][0])

            for k in clients:
                # Weighted mean values for each class
                # Calibration Mean
                mean_c += (client_length[k][c] / n_c) * \
                    np.array(client_mean[k][c])

                mean_ck = np.array(client_mean[k][c])
                mul_mean += ((client_length[k][c]) /
                             (n_c - 1)) * np.dot(mean_ck.T, mean_ck)

                cov_ck += ((client_length[k][c] - 1) /
                           (n_c - 1)) * np.array(client_cov[k][c])

            g_mean.append(mean_c)
            cov_c = cov_ck + mul_mean - \
                (n_c / (n_c - 1)) * np.dot(mean_c.T,
                                           mean_c)  # Calibration Covariance

            g_cov.append(cov_c)

        return g_mean, g_cov

    def get_feature_label(self):
        self.global_model.eval()

        cnt = 0
        features = []
        true_labels = []
        pred_labels = []
        for batch_id, batch in enumerate(self.test_loader):
            data, target = batch
            cnt += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            # if dataset is tabular then convert the target to 2d array [64,1]
            if self.conf['data_type'] == 'tabular':
                target = target.int().view(-1, 1)

            feature, output = self.global_model(data)
            # get the index of the max log-probability
            # pred = output.data.max(1)[1]
            if self.conf['classification_type']=="multi":
                pred = output.data.max(1)[1]
            elif self.conf['classification_type']=="binary":
                pred = (torch.sigmoid(output) > 0.5).int()
            features.append(feature)
            true_labels.append(target)
            pred_labels.append(pred)

            if cnt > 2000:
                # print("len(self.test_loader.dataset)",len(self.test_loader.dataset))
                break

        features = torch.cat(features, dim=0)
        true_labels = torch.cat(true_labels, dim=0)
        pred_labels = torch.cat(pred_labels, dim=0)

        return features, true_labels, pred_labels
