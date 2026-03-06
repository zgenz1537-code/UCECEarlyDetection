if True:
    from reset_random import reset_random

    reset_random()
import os

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

from model import crnn
from utils import CLASSES, plot, TrainingCallback

matplotlib.use("Qt5Agg")
plt.rcParams["font.family"] = "JetBrains Mono"

ACC_PLOT = plt.figure(num=2)
LOSS_PLOT = plt.figure(num=3)


def get_data():
    dp = "Data/features_selected.csv"
    print("[INFO] Loading Data :: {0}".format(dp))
    df = pd.read_csv(dp)
    x_, y_ = df.values[:, :-1].astype(float), df.values[:, -1].astype(int)
    return x_, y_


if __name__ == "__main__":
    ratios = [0.2, 0.3]
    for ratio in ratios:
        reset_random()

        x, y = get_data()

        print("[INFO] Normalizing Data")
        ss = StandardScaler()
        x = ss.fit_transform(x)

        y_cat = to_categorical(y, len(CLASSES))
        x = np.expand_dims(x, axis=1)
        trs, tes = int(100 - ratio * 100), int(ratio * 100)
        print(
            "[INFO] Splitting Data Into Training|Testing ==> {0}|{1}".format(trs, tes)
        )
        train_x, test_x, train_y, test_y = train_test_split(
            x, y, test_size=ratio, shuffle=True, random_state=1
        )
        test_y_cat = to_categorical(test_y, len(CLASSES))
        print("[INFO] X Shape :: {0}".format(x.shape))
        print("[INFO] Train X Shape :: {0}".format(train_x.shape))
        print("[INFO] Test X Shape :: {0}".format(test_x.shape))

        model_dir = os.path.join("models", "{0}-{1}".format(trs, tes))
        os.makedirs(model_dir, exist_ok=True)

        acc_loss_csv_path = os.path.join(model_dir, "acc_loss.csv")
        model_path = os.path.join(model_dir, "model.h5")

        training_cb = TrainingCallback(acc_loss_csv_path, ACC_PLOT, LOSS_PLOT)
        checkpoint = ModelCheckpoint(
            model_path,
            save_best_only=True,
            save_weights_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=False,
        )

        model = crnn(x.shape[1:])

        initial_epoch = 0
        if os.path.isfile(model_path) and os.path.isfile(acc_loss_csv_path):
            print("[INFO] Loading Pre-Trained Model :: {0}".format(model_path))
            model.load_weights(model_path)
            initial_epoch = len(pd.read_csv(acc_loss_csv_path))

        print("[INFO] Fitting Data")
        model.fit(
            x,
            y_cat,
            validation_data=(test_x, test_y_cat),
            batch_size=16,
            epochs=25,
            verbose=0,
            initial_epoch=initial_epoch,
            callbacks=[training_cb, checkpoint],
        )

        model.load_weights(model_path)

        train_prob = model.predict(train_x)
        train_pred = np.argmax(train_prob, axis=1).ravel().astype(int)
        plot(
            train_y.astype(int),
            train_pred,
            train_prob,
            os.path.join("{0}-{1}".format(trs, tes), "Train"),
        )

        test_prob = model.predict(test_x)
        test_pred = np.argmax(test_prob, axis=1).ravel().astype(int)
        plot(
            test_y.astype(int),
            test_pred,
            test_prob,
            os.path.join("{0}-{1}".format(trs, tes), "Test"),
        )
