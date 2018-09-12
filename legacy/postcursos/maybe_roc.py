import pandas as pn
import matplotlib.pyplot as plt
import re
import numpy as np


datar = pn.read_csv("~/Workspace/Tesis/data/census/maybe/roc/df.csv")
df = datar[datar.columns[1:]]

df["privacy"] = df["case"].map(lambda x: re.findall("\d+", x)[0])
df["real"] = df["case"].map(lambda x: re.findall("[^\d]",x)[0])
df["uniform"] = df["case"].map(lambda x: int(re.findall("[^\d]",x)[1] == "t"))


# make one roc curve for each case, also get the roc of the hierarchical levels
def rocs_by_privacy(df, real, uniform):
    #real = "t"
    #uniform = 0

    data = df
    fig, ax = plt.subplots()
    labels = []

    for pr in data.privacy.unique():
        data_f = data.query("privacy == '{pr}'".format(pr=int(pr))) if pr is not None else data
        data_f = data_f.query("real=='{real}'".format(real=real)) if real is not None else data_f
        data_f = data_f.query("uniform=={uniform}".format(uniform=uniform)) if uniform is not None else data_f

        x = data_f.fpr
        y = data_f.tpr
        lines, _ = ax.get_legend_handles_labels()
        ax.plot(x, y)
        labels.append(pr)

    ax.legend(lines, labels, loc='best')
    tt = "Income DB, real=" + str(real) + ", uniform=" + str(uniform)
    tt_save = "privacy" + str(real) + str(uniform)
    ax.set_title(tt)
    ax.set_xlabel("privacy")
    plt.savefig("plots/income_roc_" + tt_save + ".png")
    plt.show()


def rocs_gb(df, reals=["t", "m", "f"], uniforms=[0,1]):
    #real = "t"
    #uniform = 0

    data = df
    fig, ax = plt.subplots()
    labels = []

    for real in reals:
        for uniform in uniforms:
            #for pr in data.privacy.unique():
            #data_f = data.query("privacy == '{pr}'".format(pr=int(pr))) if pr is not None else data
            data_f = data.query("real=='{real}'".format(real=real)) if real is not None else data
            data_f = data_f.query("uniform=={uniform}".format(uniform=uniform)) if uniform is not None else data_f
            data_f.loc[:,"fpr_dis"] = data_f["fpr"].map(lambda x: round(x,2))

            gb = data_f.groupby("fpr_dis")["tpr"].mean().reset_index().sort_values(by="fpr_dis", ascending=True)
            x = gb.fpr_dis
            y = pn.rolling_mean(gb.tpr,3)
            ax.plot(x, y)
            lines, _ = ax.get_legend_handles_labels()

            tt = "real=" + str(real) + ", uniform=" + str(uniform)

            labels.append(tt)

    ax.legend(lines, labels, loc='best')
    tt = "ROC considering if real and distribution"
    tt_save = "privacy_grouped_" + "".join(reals) + "_" + "".join([str(x) for x in uniforms])
    ax.set_title(tt)
    ax.set_xlabel("privacy")
    plt.savefig("plots/income_roc_" + tt_save + ".png")
    plt.show()

rocs_gb(df, ["t", "m", "f"], [0, 1])
rocs_gb(df, ["t", "m", "f"], [0])
rocs_gb(df, ["t", "m", "f"], [1])

rocs_gb(df, ["t"], [0, 1])
rocs_gb(df, ["m"], [0, 1])
rocs_gb(df, ["f"], [0, 1])




for real in ["t", "m", "f"]:
    for uniform in [0, 1]:
        rocs_by_privacy(df, real, uniform)

