"""
To obtain your Kaggle API Token, visit https://www.kaggle.com/settings,
navigate to the API section, and create a new token. Download the generated
'kaggle.json' file. Replace the placeholders 'kaggle_username' and 'kaggle_api_key'
in the script with your Kaggle username and API key, respectively.
"""

##Configuration
conf = {
	"dataset_used": "kaggle", #type of dataset, mnist, cifar10, kaggle
	
	#Type of data，tabular, image
	"data_type" : "tabular",
		
	#Model selection: mlp, cnn 
	"model_name" : "mlp",
	
	#prediction method
	"classification_type": "binary", #binary or multi

	# if using binary class True, else for multi class False
	"loss_criterion_binary": True, # True, False
	
	#Classes
	"num_classes": 2, #binary = 2, #multi_class = 10

	#number of parties
	"num_parties":10, #10

	"kaggle":{
		# dataset used from kaggle
		"dataset_used": "adult_income",

		#dataset link to download {user_name}/{dataset_name}
		"kaggle_dataset_download": "wenruliu/adult-income-dataset",

		# replace kaggle_username with your username
		"kaggle_username":"",

		#replace kaggle_api_key with your token api key
		"kaggle_api_key":"",
	},

	#Data processing method: fed_ccvr
	"no-iid": "fed_ccvr",

	# client_optimizer used
    "client_optimizer": "SGD", #Adam, SGD

	#re_train_optimizer used
	"re_train_optimizer": "Adam", #Adam, SGD

	#Global epoch
	"global_epochs" :2, #100, 5

	#Local epoch
	"local_epochs" : 3,

	#dirichlet distribution
	"beta" : 0.5, #0.5 > 0.05, 0.1, 0.5, 0.9
	"batch_size" : 64,
	"weight_decay":1e-5,

    #learning rate
	"lr" : 0.005,#0.001
	"momentum" : 0.9,

    #Model aggregation
	"is_init_avg": True,

    #Local val test ratio
	"split_ratio": 0.3,

    #Label name
	"label_column": "label",

	#Data name
	"data_column": "file",

    #Test dataset , 
	"test_dataset": "./data/dataset/test/test.csv",

    #Train dataset
	"train_dataset" : "./data/dataset/train/train.csv", 

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
		"epoch": 10,
		"lr": 0.001,
		"num_vr":2000
	}
}
