import pandas as pd
import warnings
from utils import CLASSES


warnings.filterwarnings("ignore", category=Warning)


def load_data():
    data_path = "Data/Uterine Corpus Endometrial Carcinoma.csv"
    print("[INFO] Loading Data From :: {0}".format(data_path))
    df = pd.read_csv(data_path)
    print("[INFO] Data Shape :: {0}".format(df.shape))
    return df


def replace_categorical_cols(df):
    for col in df.columns:
        if col == "Tumor Type":
            continue
        if df[col].dtype == "str":
            print("[INFO] Replacing Categorical Values in Column :: {0}".format(col))
            repd = {v: k + 1 for k, v in enumerate(df[col].unique())}
            df[col] = df[col].replace(repd)
    df.reset_index(drop=True, inplace=True)
    return df


def preprocess(df):
    df.drop(["Patient ID", "Sample ID"], axis=1, inplace=True)
    df["Tumor Type"] = df["Tumor Type"].replace(
        {
            "Endometrioid Endometrial Adenocarcinoma": "Class 1",
            "Serous Endometrial Adenocarcinoma": "Class 2",
            "Mixed Serous and Endometrioid Carcinoma": "Class 3",
        }
    )
    r_d = {v: k for k, v in enumerate(CLASSES)}
    print("[INFO] {0}".format(r_d))
    df["Tumor Type"] = df["Tumor Type"].replace(r_d)
    print("[INFO] Replacing Categorical Attack Types To Numerical")
    df = replace_categorical_cols(df)
    df = df.fillna(df.mean())
    df['Tumor Type'] = df['Tumor Type'].astype(int)
    dp = "Data/preprocessed.csv"
    print("[INFO] Saving PreProcessed Data :: {0}".format(dp))
    df.to_csv(dp, index=False)
    return df


if __name__ == "__main__":
    df_ = load_data()
    df_ = preprocess(df_)
