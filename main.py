import json,os
from conf import conf
import torch
import numpy as np
from fedavg.server import Server
from fedavg.client import Client
from fedavg.models import CNN_Model,weights_init_normal, ReTrainModel,MLP,ReTrainModelCNN
from utils import get_data
import copy
import torch.nn.functional as F
import pandas as pd

# List of beta values to iterate over
beta_values = [0.3]

# Main federated learning and retraining loop for each beta value
for beta in beta_values:
    # Create a copy of the conf dictionary to avoid overwriting the original
    current_conf = copy.deepcopy(conf)

    # Set the beta value for this iteration
    current_conf["beta"] = beta

    # Modify file names to avoid overwriting files across different beta runs
    current_conf["model_file"] = f"model_beta_{beta}.pth"
    current_conf["retrain_model_file"] = f"retrained_model_beta_{beta}.pth"
    current_conf['save_epochs_info']['train_info_file'] = f"train_info_beta_{beta}.csv"
    current_conf['save_epochs_info']['re_train_info_file'] = f"re_training_info_beta_{beta}.csv"
    current_conf['save_epochs_info']['only_global_epochs_file'] = f"only_global_epochs_beta_{beta}.csv"

    # Ensure the model_dir and save_info directory exists
    if not os.path.isdir(current_conf["model_dir"]):
        os.makedirs(current_conf["model_dir"], exist_ok=True)

    save_info_dir = current_conf['save_epochs_info']['dir_name']
    if not os.path.exists(save_info_dir):
        os.makedirs(save_info_dir, exist_ok=True)

    # Initialize the datasets and clients
    train_datasets, val_datasets, test_dataset = get_data()

    # Initialization of aggregation values 
    client_weight = {}
    if current_conf["is_init_avg"]:
        for key in train_datasets.keys():
            client_weight[key] = 1 / len(train_datasets)

    print("Aggregation value initialization")

    # Save clients and models
    clients = {}
    clients_models = {}

    # Model initialization
    if current_conf['model_name'] == "mlp":
        n_input = 2352  # Or change as per the dataset
        model = MLP(n_input, 512, current_conf["num_classes"])
        print("-----------Model for MLP selected!-----------")

        if current_conf['dataset_used'] == "kaggle":
            if current_conf['kaggle']['dataset_used'] == "adult_income":
                model = MLP(n_feature=14, n_hidden=512, n_output=1)

    elif current_conf['model_name'] == 'cnn':
        model = CNN_Model()

    model.apply(weights_init_normal)

    if torch.cuda.is_available():
        model.cuda()

    # Initialize the server with the model and test dataset
    server = Server(current_conf, model, test_dataset)
    print("Server initialization finished!")

    for key in train_datasets.keys():
        clients[key] = Client(current_conf, server.global_model, train_datasets[key], val_datasets[key])

    print("Client initialization finished!")

    max_acc = 0

    global_training_info = []
    only_gloabl_train_df = []

    # Federated learning process
    for e in range(current_conf["global_epochs"]):

        for key in clients.keys():
            print(f'Training client {key} for epoch {e}...')
            model_k, local_train_info_df = clients[key].local_train(server.global_model, key, e)
            global_training_info.extend(local_train_info_df)
            clients_models[key] = copy.deepcopy(model_k)

        # Central (FL Aggregation)
        server.model_aggregate(clients_models, client_weight)

        # Test global model
        acc, loss = server.model_eval()
        global_training_info.append({
            'global_epoch': e,
            'client_id': None,
            'epoch': None,
            'train_loss': None,
            'eval_loss': None,
            'eval_acc': None,
            'global_acc': acc,
            'global_loss': loss
        })
        only_gloabl_train_df.append({
            'global_epoch': e,
            'global_acc': acc,
            'global_loss': loss
        })

        df = pd.DataFrame(only_gloabl_train_df)
        file_name = current_conf["dataset_used"] + "_" + current_conf['save_epochs_info']['only_global_epochs_file']
        file_path = os.path.join(current_conf["save_epochs_info"]['dir_name'], file_name)
        df.to_csv(file_path, mode='w', index=False)
        print(f"Epoch {e}, global_acc: {acc:.6f}, global_loss: {loss:.6f}\n")

        # Save model checkpoint after each epoch
        model_file_name = current_conf["dataset_used"] + f"_model_epoch_{e}_beta_{beta}.pth"
        torch.save(server.global_model.state_dict(), os.path.join(current_conf["model_dir"], model_file_name))

    # Save global training info after all epochs
    df = pd.DataFrame(global_training_info)
    train_file_name = current_conf["dataset_used"] + "_" + current_conf['save_epochs_info']['train_info_file']
    train_file_path = os.path.join(current_conf['save_epochs_info']['dir_name'], train_file_name)
    df.to_csv(train_file_path, mode='w', index=False)
    print(f'{train_file_path} saved successfully.')

    # Save final global model after training
    final_model_file = current_conf["dataset_used"] + "_" + current_conf["model_file"]
    torch.save(server.global_model.state_dict(), os.path.join(current_conf["model_dir"], final_model_file))
    print(f"Final model {final_model_file} saved successfully.")

    # Start retraining process
    print("Re-training started!")

if conf['no-iid'] == 'fl-fcr':
    # Use Virtual representation (T-sne)
    client_mean = {}
    client_cov = {}
    client_length = {}
    print("Mean and Covariance calculation started!")
    for key in clients.keys():
        # Calculate mean and covariance
        c_mean, c_cov, c_length = clients[key].cal_distributions(server.global_model, load_pair_data=False)
        client_mean[key] = c_mean
        client_cov[key] = c_cov
        client_length[key] = c_length
    print("Mean and Covariance calculation finished!")

    # Calculate GLOBAL mean and covariance
    g_mean, g_cov = server.cal_global_gd(client_mean, client_cov, client_length)
    print("Calculation of Global mean and covariance done!")

    # Virtual representations generation
    retrain_vr = []
    label = []
    eval_vr = []
    for i in range(conf['num_classes']):
        mean = np.squeeze(np.array(g_mean[i]))
        vr = np.random.multivariate_normal(mean, g_cov[i], conf["retrain"]["num_vr"] * 2)
        retrain_vr.extend(vr.tolist()[:conf["retrain"]["num_vr"]])
        eval_vr.extend(vr.tolist()[conf["retrain"]["num_vr"]:])
        label.extend([i] * conf["retrain"]["num_vr"])

    print("# Virtual representations done!")

    # Retrieve the model to be retrained
    if conf['model_name'] == "mlp":
        retrain_model = ReTrainModel()
    elif conf['model_name'] == "cnn":
        retrain_model = ReTrainModelCNN()

    if torch.cuda.is_available():
        retrain_model.cuda()

    reset_name = []
    for name, _ in retrain_model.state_dict().items():
        reset_name.append(name)

    # Copy global model parameters to retrain model
    for name, param in server.global_model.state_dict().items():
        if name in reset_name:
            retrain_model.state_dict()[name].copy_(param.clone())

    retrain_global_info = []
    # Use VR to re-train the model
    retrain_model, re_train_info_df = server.retrain_vr(retrain_vr, label, eval_vr, retrain_model)
    retrain_global_info.extend(re_train_info_df)
    print("Re-training done")

    # Update the global model with the newly-trained parameters
    for name, param in retrain_model.state_dict().items():
        server.global_model.state_dict()[name].copy_(param.clone())

    # Evaluate the retrained model
    acc, loss = server.model_eval()
    retrain_global_info.append({
        'retraining_epoch': None,
        'train_loss': None,
        'eval_loss': None,
        'eval_acc': None,
        'retraining_global_acc': acc,
        'retraining_global_loss': loss
    })

    print(f"After retraining global_acc: {acc}, global_loss: {loss}")
    df = pd.DataFrame(retrain_global_info)

    # Specify the file path for re-training info
    file_name = conf["dataset_used"] + "_" + conf['save_epochs_info']['re_train_info_file']
    file_path = os.path.join(conf["save_epochs_info"]['dir_name'], file_name)


    df.to_csv(file_path, index=False)
    print(f'{file_path} saved successfully.')

    # Correct the file naming convention for saving the retrained model
    # model_file_name = f"{conf['dataset_used']}_retrained_model_beta_{conf['beta']}.pth"
    # torch.save(server.global_model.state_dict(), os.path.join(conf["model_dir"], model_file_name))

    model_file_name = f"{current_conf['dataset_used']}_retrained_model_beta_{current_conf['beta']}.pth"
    torch.save(server.global_model.state_dict(), os.path.join(current_conf["model_dir"], model_file_name))

    
    print(f"Re-trained global model file saved at: {model_file_name}")
    print(f"FL done, model has been saved under {conf['model_dir']}!")
    
