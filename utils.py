import os

import matplotlib
import numpy as np
import pandas as pd
import prettytable
from matplotlib import pyplot as plt
from pycm import ConfusionMatrix
from keras.callbacks import Callback

from performance_evaluator.plots import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

matplotlib.use("Qt5Agg")
plt.rcParams["font.family"] = "JetBrains Mono"

CLASSES = ["Class 1", "Class 2", "Class 3"]


def print_df_to_table(df, p=True):
    field_names = list(df.columns)
    p_table = prettytable.PrettyTable(field_names=field_names)
    p_table.add_rows(df.values.tolist())
    d = "\n".join(
        ["\t\t{0}".format(p_) for p_ in p_table.get_string().splitlines(keepends=False)]
    )
    if p:
        print(d)
    return d


class TrainingCallback(Callback):
    def __init__(self, acc_loss_path, plt1, plt2):
        self.acc_loss_path = acc_loss_path
        self.plt1 = plt1
        self.plt2 = plt2
        if os.path.isfile(self.acc_loss_path):
            self.df = pd.read_csv(self.acc_loss_path)
            plot_acc_loss(
                self.df, self.plt1, self.plt2, os.path.dirname(self.acc_loss_path)
            )
        else:
            self.df = pd.DataFrame(
                [], columns=["epoch", "accuracy", "val_accuracy", "loss", "val_loss"]
            )
            self.df.to_csv(self.acc_loss_path, index=False)
        Callback.__init__(self)

    def on_epoch_end(self, epoch, logs=None):
        self.df.loc[len(self.df.index)] = [
            int(epoch + 1),
            round(logs["accuracy"], 4),
            round(logs["val_accuracy"], 4),
            round(logs["loss"], 4),
            round(logs["val_loss"], 4),
        ]
        self.df.to_csv(self.acc_loss_path, index=False)
        print(
            "[EPOCH :: {0}] -> Acc :: {1} | Val_Acc :: {2} | Loss :: {3} | Val_Loss :: {4}".format(
                epoch + 1, *[format(v, ".4f") for v in self.df.values[-1][1:]]
            )
        )
        plot_acc_loss(
            self.df, self.plt1, self.plt2, os.path.dirname(self.acc_loss_path)
        )


def plot_line(plt_, y1, y2, epochs, for_, save_path):
    ax = plt_.gca()
    ax.clear()
    ax.plot(range(epochs), y1, label="Training", color="dodgerblue")
    ax.plot(range(epochs), y2, label="Validation", color="orange")
    ax.set_title("Training and Validation {0}".format(for_))
    ax.set_xlabel("Epochs")
    ax.set_ylabel(for_)
    ax.set_xlim([0, epochs])
    ax.legend()
    plt_.tight_layout()
    plt_.savefig(save_path)


def plot_acc_loss(df, plt1, plt2, save_dir):
    epochs = len(df)
    acc = df["accuracy"].values
    val_acc = df["val_accuracy"].values
    loss = df["loss"].values
    val_loss = df["val_loss"].values
    plot_line(
        plt1, acc, val_acc, epochs, "Accuracy", os.path.join(save_dir, "accuracy.png")
    )
    plot_line(plt2, loss, val_loss, epochs, "Loss", os.path.join(save_dir, "loss.png"))


def plot(y, pred, prob, for_):
    print("[INFO] Evaluating {0} Data".format(for_))
    results_dir = "results/{0}".format(for_)
    os.makedirs(results_dir, exist_ok=True)

    classes = CLASSES
    cm = ConfusionMatrix(actual_vector=y, predict_vector=pred)
    measures = {
        "ACC": ["ACC", "ACC Macro"],
        "Sensitivity": ["TPR", "TPR Micro"],
        "Specificity": ["TNR", "TNR Micro"],
        "F1-Score": ["F1", "F1 Micro"],
    }
    results_dict = {"Class": list(CLASSES) + ["Overall"]}
    # cm_arr = cm.to_array()
    # acc = list([diag / np.sum(cm_arr[i]) for i, diag in enumerate(cm_arr.diagonal())])
    # overall_acc = sum(acc) / len(acc)
    # results_dict['Accuracy'] = list(map(lambda val: round(0.0 if val == 'None' else val, 4), acc + [overall_acc]))
    for k in measures.keys():
        v = list(cm.class_stat[measures[k][0]].values())
        v.append(cm.overall_stat[measures[k][1]])
        v = list(map(lambda val: round(0.0 if val == "None" else val, 4), v))
        results_dict[k] = v
    mcc = list(cm.class_stat["MCC"].values())
    mcc = mcc + [sum(mcc) / len(CLASSES)]
    results_dict["MCC"] = list(
        map(lambda val: round(0.0 if val == "None" else val, 4), mcc)
    )
    df = pd.DataFrame.from_dict(results_dict)
    df.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)
    print_df_to_table(df)

    plt.figure()
    ax = plt.gca()
    confusion_matrix(
        y, pred, classes, ax=ax, xticklabels_rotation=90, yticklabels_rotation=0
    )
    plt.savefig(os.path.join(results_dir, "conf_mat.png"))
    plt.clf()
    plt.cla()
    plt.close()

    plt.figure()
    ax = plt.gca()
    precision_recall_curve(y, prob, classes, ax=ax, legend_ncol=1)
    plt.savefig(os.path.join(results_dir, "pr_curve.png"))
    plt.clf()
    plt.cla()
    plt.close()

    plt.figure()
    ax = plt.gca()
    roc_curve(y, prob, classes, ax=ax, legend_ncol=1)
    plt.savefig(os.path.join(results_dir, "roc_curve.png"))
    plt.clf()
    plt.cla()
    plt.close()
