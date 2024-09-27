import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os


def adult_income_dataset():
    DATASET_PATH = "./data/dataset/adult.csv"
    dfx = pd.read_csv(DATASET_PATH)

    TARGET_COLUMN = "income"

    numeric_columns = [
        (col, preprocessing.StandardScaler())
        for col in dfx.columns
        if dfx[col].dtype == "int64" and col != TARGET_COLUMN
    ]
    categorical_columns = [
        (col, preprocessing.LabelEncoder())
        for col in dfx.columns
        if dfx[col].dtype == "object" and col != TARGET_COLUMN
    ]

    for col, sclr in numeric_columns:
        dfx.loc[:, col] = sclr.fit_transform(dfx[[col]])

    for col, lbl_enc in categorical_columns:
        dfx[col] = lbl_enc.fit_transform(dfx[col].values)

    target_encoding = {"<=50K": 0, ">50K": 1}
    dfx.loc[:, TARGET_COLUMN] = dfx[TARGET_COLUMN].map(target_encoding)

    df_updated = dfx.rename(columns={'income': 'label'})

    train_df, test_df = train_test_split(df_updated, test_size = 0.2)

    #  Specify the directories for training and testing data
    train_dir = './data/dataset/train/'
    test_dir = './data/dataset/test/'

    # Create the directories if they do not exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Save training and testing DataFrames to CSV files without including the index
    train_df.to_csv(os.path.join(train_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(test_dir, 'test.csv'), index=False)










