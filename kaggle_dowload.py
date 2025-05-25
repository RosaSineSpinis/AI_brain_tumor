import os
import kagglehub

# Set up Kaggle API key environment variable
os.environ['KAGGLE_CONFIG_DIR'] = '/Users/Piotr/.kaggle/kaggle.json'

# Use Kaggle API to download the dataset
path = kagglehub.dataset_download("awsaf49/brats20-dataset-training-validation")

print("Path to dataset files:", path)

# !kaggle datasets download -d awsaf49/brats20-dataset-training-validation -p /path/to/save/dataset --unzip