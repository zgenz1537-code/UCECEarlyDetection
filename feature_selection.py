if True:
    from reset_random import reset_random

    reset_random()
import os

import numpy as np
import pandas as pd


from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def fast_mrmr(X, y, k=6):

    # relevance between feature and target
    relevance = mutual_info_classif(X, y)

    selected = []
    remaining = list(range(X.shape[1]))

    # select first feature with highest relevance
    first = np.argmax(relevance)
    selected.append(first)
    remaining.remove(first)

    while len(selected) < k and remaining:

        scores = []

        for f in remaining:

            redundancy = np.mean(
                [mutual_info_regression(X[:, [f]], X[:, s])[0] for s in selected]
            )

            score = relevance[f] - redundancy
            scores.append(score)

        best_feature = remaining[np.argmax(scores)]

        selected.append(best_feature)
        remaining.remove(best_feature)

    return selected


def save_data(df, cols, save_dir):
    columns = list(df.columns)
    selected_cols = []
    for col in cols:
        selected_cols.append(columns[col])
    print("[INFO] Total Features :: {0}".format(len(df.columns) - 1))
    print("[INFO] Selected Features :: {0}".format(len(selected_cols)))
    with open(os.path.join(save_dir, "selected_features.txt"), "w") as f:
        f.write("\n".join(selected_cols))
    selected_cols.extend(["Tumor Type"])
    sp = os.path.join(save_dir, "features_selected.csv")
    print("[INFO] Saving Feature Selected Data To File :: {0}".format(sp))
    df[selected_cols].to_csv(sp, index=False)
    return df[selected_cols]


def select_features(df):
    reset_random()
    sd = "Data"
    os.makedirs(sd, exist_ok=True)

    df_ = df.copy(deep=True)

    X = df.values[:, :-1].astype(float)
    y = df.values[:, -1].astype(int)

    print("[INFO] Feature Selection Using Fast mRMR")

    selected_feats = fast_mrmr(X, y, k=6)

    return save_data(df_, selected_feats, sd)


if __name__ == "__main__":
    select_features(pd.read_csv("Data/preprocessed.csv"))
