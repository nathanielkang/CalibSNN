"""
To obtain your Kaggle API Token, visit https://www.kaggle.com/settings,
navigate to the API section, and create a new token. Download the generated
'kaggle.json' file. Replace the placeholders 'kaggle_username' and 'kaggle_api_key'
in the script with your Kaggle username and API key, respectively.
"""

# note: run cross entropy: (1) 'retrain_loss_criterion':2, (2) "train_load_data":False, (3) "train_contrastive_learning":False, (4)"train_loss_criterion": 2
# note: CalibSNN: (1) 'retrain_loss_criterion':4, (2) "train_load_data":False, (3) "train_contrastive_learning":False, (4)"train_loss_criterion": 4


##Configuration
conf = {
	"dataset_used": "cifar10", #type of dataset, mnist, cifar10, usps, cifar100, fmnist, synthetic
	
	# 1: BinaryCrossEntropy
	# 2: CategoricalCrossEntropy,
	# 3: SNN Loss only
	# 4: CalibSNN Loss (CE + SNN)

	'retrain_loss_criterion': 4,
 
	# Run for Client 
	"train_load_data":False, # Training Dataset loaded in Pairs for old contrastive learning
	"train_contrastive_learning":False, # Use old contrastive loss (deprecated)
	"train_loss_criterion": 4,  # 4: CalibSNN Loss

	## Data Load and implement Loss 
	"eval_load_data":False, # Evaluation Dataset loaded in Pairs
	"eval_contrastive_learning":False,
	"eval_loss_criterion": 4,  # 4: CalibSNN Loss

	# Run for Server
	"test_load_data":False, # Load Test dataset in pairs # to run `visualize.py` put it False
	"test_contrastive_learning":False, # Implement Contrastive Loss
	"test_loss_criterion": 2, # Use CE for testing

	#Type of data，tabular, image
	"data_type" : "image",
		
	#Model selection: mlp, cnn 
	"model_name" : "cnn",
	
	#prediction method
	"classification_type": "multi", #binary or multi

	# if using binary class True, else for multi class False
	# "loss_criterion_binary": True, # True, False
	
	# Temperature parameter for SNN loss (increased for stability)
	'--tau': 2.5,
	
	# Weight for SNN loss component in CalibSNN (reduced for stability)
	'lambda_snn': 0.5,
	
	# CalibSNN specific parameters
	'calibsnn': {
		'enable_calibration': True,  # Enable feature calibration
		'enable_resampling': True,   # Enable re-sampling during training
		'resample_ratio': 0.2,       # Ratio of synthetic samples to add
		'update_global_every': 5,    # Update global statistics every N rounds
	},
	
	#Classes
	"num_classes": 10, #binary = 2, #multi_class = 10, # cifar100 = 100

	#number of parties
	"num_parties":20, 

	"kaggle":{
		# dataset used from kaggle
		"dataset_used": "adult_income",

		#dataset link to download {user_name}/{dataset_name}
		"kaggle_dataset_download": "wenruliu/adult-income-dataset",

		# replace kaggle_username with your username
		"kaggle_username":"bilalahmadai",

		#replace kaggle_api_key with your token api key
		"kaggle_api_key":"d0467d263f404eac2e2752895eef4b07",
	},

	#Data processing method: CalibSNN
	"no-iid": "calibsnn",

	# client_optimizer used
    "client_optimizer": "SGD", #Adam, SGD 

	#re_train_optimizer used
	"re_train_optimizer": "SGD", #Adam, SGD

	#Global epoch
	"global_epochs" : 50, 

	#Local epoch
	"local_epochs" : 10,

	#dirichlet distribution
	"beta" : 0.05, 
	"batch_size" : 1024,
	"weight_decay":1e-6,

    #learning rate
	"lr" : 0.01,
	"momentum" : 0.9,

    #Model aggregation
	"is_init_avg": True,

    #Local val test ratio
	"split_ratio": 0.2,
 
	# Synthetic Data Generation
    "gamma": 0.8,                 # Controls the distribution of weights (adjustable)
    "iid": False,                  # Set True for IID data, False for non-IID data
    "dimension": 3072,            # Feature dimension (e.g., 32x32x3 = 3072 for CIFAR-like data)
    # You can define additional parameters if needed for different settings
    "synthetic_samples_mean": 5,  # Mean for lognormal sample generation
    "synthetic_samples_std": 3,   # Standard deviation for lognormal sample generation

    #Label name
	"label_column": "label",

	#Data name
	"data_column": "file",

    # Test dataset , 
	"test_dataset": "../data/dataset/test/test.csv",

    #Train dataset
	"train_dataset" : "../data/dataset/train/train.csv", 

    #Where to save the model:
	"model_dir":"./save_model/",

    #Model name:
	"model_file":"model.pth",
	#Retrained Model name:
	"retrain_model_file":"retrained_model.pth",

	#save training epoch info
	"save_epochs_info" :{
		# make dir to save info in csv
		"dir_name" : "./save_info/",
		# for training_server_&_client epochs .csv file 
		"train_info_file" :"train_info.csv",
		# for re_training epochs .csv file 
		"re_train_info_file" :"re_training_info.csv",
		# for re_training epochs .csv file 
		"only_global_epochs_file" :"only_global_epochs.csv",
	},

	"retrain":{
		"epoch": 100,
		"lr": 0.01,
		"weight_decay": 1e-6,
		"num_vr":2000
	},

	
}

# Dataset configurations
dataset_configs = {
    "mnist": {
        "data_type": "image",
        "num_classes": 10,
        "data_path": "../data/dataset/mnist",
        "train_path": "../data/dataset/mnist/train",
        "test_path": "../data/dataset/mnist/test"
    },
    "cifar10": {
        "data_type": "image", 
        "num_classes": 10,
        "data_path": "../data/dataset/cifar10",
        "train_path": "../data/dataset/cifar10/train",
        "test_path": "../data/dataset/cifar10/test"
    },
    "usps": {
        "data_type": "image",
        "num_classes": 10,
        "data_path": "../data/dataset/usps",
        "train_path": "../data/dataset/usps/train",
        "test_path": "../data/dataset/usps/test"
    },
    "svhn": {
        "data_type": "image",
        "num_classes": 10,
        "data_path": "../data/dataset/svhn",
        "train_path": "../data/dataset/svhn/train",
        "test_path": "../data/dataset/svhn/test"
    },
    "kddcup99": {
        "data_type": "tabular",
        "num_classes": 2,
        "data_path": "../data/dataset/kddcup99",
        "train_path": "../data/dataset/kddcup99/train",
        "test_path": "../data/dataset/kddcup99/test"
    },
    "ag_news": {
        "data_type": "text",
        "num_classes": 4,
        "data_path": "../data/dataset/ag_news",
        "train_path": "../data/dataset/ag_news/train",
        "test_path": "../data/dataset/ag_news/test",
        "vocab_path": "../data/dataset/ag_news/vocab.txt"
    },
    "adult": {
        "data_type": "tabular",
        "num_classes": 2,
        "num_features": 14,  # Adult dataset has 14 features
        "data_path": "../data/dataset/adult",
        "train_path": "../data/dataset/adult/train",
        "test_path": "../data/dataset/adult/test"
    },
    "covertype": {
        "data_type": "tabular",
        "num_classes": 7,  # CoverType has 7 forest cover types
        "num_features": 54,  # CoverType has 54 features
        "data_path": "../data/dataset/covertype",
        "train_path": "../data/dataset/covertype/train",
        "test_path": "../data/dataset/covertype/test"
    }
}

# Update conf with dataset-specific settings when needed
def update_conf_for_dataset(dataset_name):
    """Update conf with dataset-specific settings."""
    if dataset_name in dataset_configs:
        config = dataset_configs[dataset_name]
        conf["data_type"] = config["data_type"]
        conf["num_classes"] = config["num_classes"]
        conf["data_path"] = config["data_path"]
        conf["train_path"] = config["train_path"]
        conf["test_path"] = config["test_path"]
        
        # Update train and test dataset paths
        conf["train_dataset"] = f"{config['train_path']}/train.csv"
        conf["test_dataset"] = f"{config['test_path']}/test.csv"
        
        if "vocab_path" in config:
            conf["vocab_path"] = config["vocab_path"]
            
        if "num_features" in config:
            conf["num_features"] = config["num_features"]