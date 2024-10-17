import os
import subprocess

# os.environ['KAGGLE_CONFIG_DIR'] = "./kaggle_token"
# Set Kaggle username and key

os.environ['KAGGLE_USERNAME'] = ""  
os.environ['KAGGLE_KEY'] = "" 
# Specify the dataset name 
dataset_name = "wenruliu/adult-income-dataset"

# Specify the directory where you want to download the dataset
download_dir = "./dataset"

# Run the Kaggle CLI command to download the dataset
command = f"kaggle datasets download -d {dataset_name} -p {download_dir} --unzip"
subprocess.run(command, shell=True)

print(f"Dataset '{dataset_name}' downloaded successfully to '{download_dir}'.")
