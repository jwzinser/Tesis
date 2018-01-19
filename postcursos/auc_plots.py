

# line plots of supervised rmse by privacy level and casse
import pandas as pn
import re
import math
df = pn.read_csv("~/Workspace/Tesis/data/census/maybe/roc/df_auc.csv")
df["privacy"] = df["case"].map(lambda x: re.findall("\d+", x)[0])
df["real"] = df["case"].map(lambda x: re.findall("[^\d]",x)[0])
df["uniform"] = df["case"].map(lambda x: int(re.findall("[^\d]",x)[1] == "t"))



def sup_auc_plot(data, gb_param, reals, uniforms):
    fig, ax = plt.subplots()
    labels = []
    for real in reals:
        for uniform in uniforms:
            data_f = data.query("real=='{real}'".format(real=real)) if real is not None else data
            data_f = data_f.query("uniform=={uniform}".format(uniform=uniform)) if uniform is not None else data_f
            data_f[gb_param] = data_f[gb_param].map(int)
            gb = data_f.sort_values(by="privacy", ascending=True)
            print(gb)
            x = gb[gb_param]
            y = gb["auc"]
            ax.plot(x, y)
            lines, _ = ax.get_legend_handles_labels()
            tt = "real=" + str(real) + ", uniform=" + str(uniform)

            labels.append(tt)

    ax.legend(lines, labels, loc='best')
    tt = "Supervised AUC real=" + str(real) + ", uniform="+str(uniform)
    tt_save = "supervised_auc_gb_" + "".join(reals) + "_" + "".join([str(x) for x in uniforms])
    ax.set_title(tt)
    ax.set_xlabel(gb_param)
    plt.savefig("plots/" + tt_save+".png")
    plt.show()

sup_auc_plot(df, "privacy", ["t", "m", "f"], [0, 1])
sup_auc_plot(df,  "privacy", ["t", "m", "f"], [0])
sup_auc_plot(df, "privacy", ["t", "m", "f"], [1])

sup_auc_plot(df, "privacy", ["t"], [0, 1])
sup_auc_plot(df, "privacy", ["m"], [0, 1])
sup_auc_plot(df,  "privacy", ["f"], [0, 1])

