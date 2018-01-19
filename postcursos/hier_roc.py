import pandas as pn
import matplotlib.pyplot as plt
import re
import numpy as np


datar = pn.read_csv("~/Workspace/Tesis/data/census/maybe/roc/df_hier.csv")
df = datar[datar.columns[1:]]



# make one roc curve for each case, also get the roc of the hierarchical levels
    #real = "t"
    #uniform = 0

data = df
fig, ax = plt.subplots()
labels = []

for pr in data.case.unique():
    if pr!=3:
        data_f = data.query("case == '{pr}'".format(pr=int(pr))) if pr is not None else data

        x = data_f.fpr
        y = data_f.tpr
        ax.plot(x, y)
        lines, _ = ax.get_legend_handles_labels()
        labels.append(pr)

ax.legend(lines, labels, loc='best')
tt = "Income DB, by hierarchy level"
ax.set_title(tt)
ax.set_xlabel("fpr")
plt.savefig("plots/income_roc_hier.png")
plt.show()
